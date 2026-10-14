"""Sensor readiness and map save/display operations."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import yaml

from onboard_runtime import (
    ROOT, RVIZ, environment, mapping_session, start, stop, write_atomic,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def level_quaternion(up):
    if len(up) != 3 or not all(math.isfinite(v) for v in up):
        raise ValueError("The map's upward vector must contain three finite values.")
    length = math.sqrt(sum(v * v for v in up))
    if length < 1e-9:
        raise ValueError("The map's upward vector cannot be zero.")
    x, y, z = [v / length for v in up]
    # Shortest rotation from measured up to +Z; no planar constraint or cloud fit.
    if z < -1 + 1e-10:
        return [1., 0., 0., 0.]
    quaternion = [y, -x, 0., 1. + z]
    norm = math.sqrt(sum(v * v for v in quaternion))
    return [v / norm for v in quaternion]



def sensor_probe(profile):
    # This branch runs only inside the selected ROS environment. Raw callbacks
    # avoid constructing thousands of Python point objects per Livox scan.
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, PointCloud2
    common = yaml.safe_load(Path(profile["config"]).read_text())["common"]
    if profile["platform"] == "agx":
        from livox_ros_driver2.msg import CustomMsg
        cloud_type, cloud_topic = CustomMsg, common["livox_lidar_topic"]
    else:
        cloud_type, cloud_topic = PointCloud2, common["lidar_topic"]
    rclpy.init()
    node = rclpy.create_node("lightning_sensor_check_" + str(os.getpid()))
    counts = dict(lidar=0, imu=0)
    acceleration = []
    def callback(key):
        def receive(_):
            counts[key] += 1
        return receive
    qos = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(cloud_type, cloud_topic, callback("lidar"), qos, raw=True)
    def receive_imu(message):
        counts["imu"] += 1
        a = message.linear_acceleration
        if all(math.isfinite(v) for v in (a.x, a.y, a.z)):
            acceleration.append((a.x, a.y, a.z))
    node.create_subscription(Imu, common["imu_topic"], receive_imu, qos)
    until = time.monotonic() + 3
    try:
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.1)
        result = dict(counts=counts, topics=[cloud_topic, common["imu_topic"]],
                      publishers=[node.count_publishers(t) for t in [cloud_topic, common["imu_topic"]]],
                      nodes=node.get_node_names())
        if len(acceleration) >= 20:
            mean = [sum(row[i] for row in acceleration) / len(acceleration) for i in range(3)]
            magnitude = math.sqrt(sum(v * v for v in mean))
            variance = sum(sum((row[i] - mean[i]) ** 2 for i in range(3))
                           for row in acceleration) / len(acceleration)
            if magnitude > 1e-6 and math.sqrt(variance) / magnitude < .05:
                result["imu_reference"] = dict(up_in_map=[v / magnitude for v in mean],
                                               samples=len(acceleration),
                                               relative_acceleration_std=math.sqrt(variance) / magnitude)
        print(json.dumps(result))
    finally:
        node.destroy_node()
        rclpy.shutdown()



def check_sensors(args, profile):
    process = start([sys.executable, str(ROOT / "scripts/onboard.py"), "--settings", str(args.settings),
                     "_probe"], profile, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = process.communicate(timeout=15)
        if process.returncode:
            raise ValueError("Sensor check failed: " + err.strip())
        try:
            result = json.loads(out)
        except json.JSONDecodeError as error:
            raise ValueError("Sensor check returned invalid JSON. "
                             "stdout: " + repr(out[:1000]) + "; stderr: " + repr(err[:1000])) from error
        return result
    finally:
        stop(process)



def report_status(observed, profile):
    for index, (key, label) in enumerate((("lidar", "LiDAR"), ("imu", "IMU"))):
        topic = observed["topics"][index]
        count = observed["counts"][key]
        if count:
            print(f"OK: {label} ({topic}): received {count} messages.")
        else:
            reason = ("publisher found, but no messages received" if observed["publishers"][index]
                      else "no publisher found and no messages received")
            print(f"ERROR: {label} ({topic}): {reason}.")
    if not all(observed["counts"].values()):
        print("NOT READY: Both LiDAR and IMU streams are required.")
        if profile["platform"] == "m20":
            print("Check multicast-relay.service on NOS and the M20 firmware sensor services.")
        elif not any(observed["publishers"]):
            print("Start the Livox driver: python3 scripts/onboard.py lidar")
        else:
            print("Check the running Livox driver and its Ethernet connection.")
        if any(not observed["counts"][key] and observed["publishers"][index]
               for index, key in enumerate(("lidar", "imu"))):
            print("Check ROS domain and DDS settings: discovery alone does not confirm delivery.")
        return
    running = [label for node, label in (("lightning_slam", "mapping"),
                                        ("lightning_localization", "localization"))
               if node in observed["nodes"]]
    if running:
        print("SUCCESS: Both sensor streams are arriving.")
        print("RUNNING: " + ", ".join(running) + ". Stop it before starting another algorithm.")
    else:
        print("SUCCESS: Sensors are ready. You can start mapping or localization.")



def save_map(settings, map_id):
    import rclpy
    from lightning.srv import SaveMap
    rclpy.init()
    node = rclpy.create_node("lightning_map_save_" + str(os.getpid()))
    session = mapping_session(settings)
    try:
        client = node.create_client(SaveMap, "/lightning/save_map")
        if not client.wait_for_service(timeout_sec=5):
            raise ValueError("No mapping save service. Start mapping before saving its map.")
        request = SaveMap.Request()
        request.map_id = map_id
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=120)
        if not future.done():
            raise ValueError("Map save is still pending. Check the mapping terminal before stopping it.")
        response = future.result().response
        if response != 0:
            raise ValueError("Map save failed (response " + str(response) + "). Check the mapping "
                             "terminal and use a new map ID.")
        print("Map saved (response: 0):", ROOT / "data" / map_id)
        if session and session.get("imu_reference") and mapping_session(settings) == session:
            directory = ROOT / "data" / map_id
            metadata = dict(version=1, source="stationary IMU samples before mapping",
                            pcd_sha256=file_hash(directory / "global.pcd"), **session["imu_reference"])
            write_atomic(directory / "map_view.json", json.dumps(metadata, indent=2) + "\n")
            print("Saved the map's gravity reference for RViz leveling.")
        else:
            print("No stationary startup gravity reference; RViz will use the original map frame.")
    finally:
        node.destroy_node()
        rclpy.shutdown()



def view_map(args, profile):
    pcd = args.map.expanduser().resolve()
    if pcd.is_dir():
        pcd /= "global.pcd"
    if not pcd.is_file():
        raise ValueError("Saved map does not exist: " + str(pcd))
    up = args.up
    reference = pcd.parent / "map_view.json"
    if not args.original_frame and up is None and reference.is_file():
        saved = json.loads(reference.read_text())
        if saved.get("version") != 1 or saved.get("pcd_sha256") != file_hash(pcd):
            raise ValueError("The saved gravity reference does not match this PCD. "
                             "Use --original-frame to inspect the original coordinates.")
        up = saved["up_in_map"]
    rotation = level_quaternion(up) if up is not None else [0., 0., 0., 1.]
    environment(profile, viewer=True)  # Diagnose missing X11 before loading a map.
    with tempfile.TemporaryDirectory(prefix="lightning-map-") as directory:
        directory = Path(directory)
        metadata = directory / "map.json"
        topic = "/lightning/saved_map_" + str(os.getpid())
        frame_topic = "/lightning/saved_map_tf_" + str(os.getpid())
        publisher = start(["ros2", "run", "lightning", "publish_map", "--ros-args",
                           "-p", "pcd_file:=" + str(pcd), "-p", "info_file:=" + str(metadata),
                           "-p", "max_points:=" + str(args.max_points),
                           "-p", "display_quaternion:=" + json.dumps(rotation),
                           "-r", "/lightning/saved_map:=" + topic,
                           "-r", "/lightning/saved_map_tf:=" + frame_topic], profile)
        viewer = None
        try:
            until = time.monotonic() + 60
            while not metadata.exists():
                if publisher.poll() is not None:
                    raise ValueError("Map publisher failed. Check the file and rebuild with scripts/build_robot.sh.")
                if time.monotonic() > until:
                    raise ValueError("Timed out loading the saved map.")
                time.sleep(.1)
            info = json.loads(metadata.read_text())
            config = yaml.safe_load((ROOT / "config/saved_map.rviz").read_text())
            manager = config["Visualization Manager"]
            manager["Displays"][1]["Topic"]["Value"] = topic
            view = manager["Views"]["Current"]
            view["Focal Point"] = dict(zip(["X", "Y", "Z"], info["center"]))
            view["Distance"] = max(5., info["radius"] * 2.5)
            rviz_config = directory / "map.rviz"
            rviz_config.write_text(yaml.safe_dump(config, sort_keys=False))
            viewer = start([RVIZ, "-d", str(rviz_config), "--ros-args",
                            "-r", "/tf:=/lightning/tf", "-r", "/tf_static:=" + frame_topic],
                           profile, viewer=True, cpus=profile.get("rviz_cpus"))
            while viewer.poll() is None:
                if publisher.poll() is not None:
                    raise ValueError("Map publisher exited while RViz was running.")
                time.sleep(.2)
            return viewer.returncode
        except KeyboardInterrupt:
            return 130
        finally:
            # A dead SSH terminal can make RViz exit reporting raise EIO.
            # Always release the publisher, even if viewer cleanup fails.
            try:
                if viewer:
                    stop(viewer)
            finally:
                stop(publisher)

