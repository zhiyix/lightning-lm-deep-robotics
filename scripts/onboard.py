#!/usr/bin/env python3
"""Configure this robot once; load its ROS environment for every onboard command."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import yaml

from onboard_runtime import (
    ROOT, RVIZ, configure, load_profile, mapping_session, process_identity,
    start, wait, write_atomic,
)
from onboard_operations import (
    check_sensors, report_status, save_map, sensor_probe, view_map,
)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--settings" and sys.argv[3] == "_probe":
        sensor_probe(load_profile(Path(sys.argv[2])))
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "--settings" and sys.argv[3] == "_save":
        save_map(Path(sys.argv[2]), sys.argv[4])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT / "data/onboard.json")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("configure", help="save the robot settings once")
    config.add_argument("platform", choices=["agx", "m20"])
    config.add_argument("--driver-workspace", type=str)
    config.add_argument("--driver-model", choices=["mid360", "mid360s"])
    config.add_argument("--config", type=Path, help="copy a calibrated sensor YAML instead of the default")
    config.add_argument("--domain", type=int)
    config.add_argument("--app-cpus")
    config.add_argument("--rviz-cpus")
    commands.add_parser("lidar", help="start the Livox driver unless it is already publishing")
    slam = commands.add_parser("slam", help="start live mapping with RViz outputs")
    slam.add_argument("--record-bag", action="store_true",
                      help="record synchronized SLAM inputs beside the session log for debug replay")
    loc = commands.add_parser("localize", help="localize against a saved tiled map")
    loc.add_argument("map", type=Path)
    loc.add_argument("--trajectory", type=Path, default=ROOT / "data/localization_onboard.tum")
    loc.add_argument("--global-init", action="store_true",
                     help="find the initial location using a map saved with a place index")
    for application in (slam, loc):
        application.add_argument("--console", choices=("auto", "plain", "verbose"), default="auto",
                                 help="live terminal status, plain summaries, or full diagnostic output")
    save = commands.add_parser("save", help="save the running mapping session")
    save.add_argument("map_id")
    rviz = commands.add_parser("rviz", help="view the live scan/path, or a saved map, over SSH X11")
    rviz.add_argument("--map", type=Path, help="saved map directory or PCD file")
    rviz.add_argument("--max-points", type=int, default=500000)
    alignment = rviz.add_mutually_exclusive_group()
    alignment.add_argument("--original-frame", action="store_true", help="ignore saved-map leveling metadata")
    alignment.add_argument("--up", type=float, nargs=3, help="measured upward direction in an older map's frame")
    status = commands.add_parser("status", help="report sensor readiness and missing inputs")
    status.add_argument("--json", action="store_true", help="print the raw sensor report for scripts")
    args = parser.parse_args()
    args.settings = args.settings.expanduser().resolve()
    if args.command == "configure":
        configure(args)
        return 0
    profile = load_profile(args.settings)
    if args.command == "rviz":
        if args.map:
            if not 1 <= args.max_points <= 500000:
                raise ValueError("--max-points must be between 1 and 500000.")
            return view_map(args, profile)
        if args.original_frame or args.up is not None:
            raise ValueError("Saved-map alignment options require --map.")
        return wait(start([RVIZ, "-d", str(ROOT / "config/onboard.rviz"), "--ros-args",
                           "-r", "/tf:=/lightning/tf", "-r", "/tf_static:=/lightning/tf_static"],
                          profile, viewer=True, cpus=profile.get("rviz_cpus")))
    if args.command == "save":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.map_id):
            raise ValueError("Map IDs may contain letters, numbers, underscores, dots and hyphens.")
        return wait(start([sys.executable, str(Path(__file__).resolve()), "--settings", str(args.settings),
                           "_save", args.map_id], profile))
    if args.command == "localize" and not (args.map.expanduser().resolve() / "index.txt").is_file():
        raise ValueError("Localization needs a complete map directory containing index.txt and its tiles.")
    if args.command == "lidar" and profile["platform"] != "agx":
        raise ValueError("M20 uses firmware drivers. Start the NOS point-cloud relay as described in section 6.1.")
    # Keep the driver lock in the supervisor, not the sourced ROS child.
    with (args.settings.parent / ".onboard-driver.lock").open("a") as lock:
        if args.command == "lidar":
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                observed = check_sensors(args, profile)
                if all(observed["counts"].values()):
                    print("LiDAR and internal IMU are already publishing; keep that driver running.")
                    return 0
                raise ValueError("The driver command is already running but both streams are not arriving. "
                                 "Check its terminal and Ethernet configuration.")
        if args.command == "status" and not args.json:
            robot = "M20 Pro" if profile["platform"] == "m20" else "Lite3 EDU / AGX"
            print(f"Checking {robot} sensor inputs (ROS domain {profile['domain']})...", flush=True)
        observed = check_sensors(args, profile)
        ready = all(observed["counts"].values())
        if args.command == "status":
            if args.json:
                print(json.dumps(observed, indent=2))
            else:
                report_status(observed, profile)
            return 0 if ready else 1
        if args.command == "lidar":
            if ready:
                print("LiDAR and internal IMU are already publishing; keep that driver running.")
                return 0
            if any(observed["publishers"]):
                raise ValueError("A sensor publisher already exists but both streams are not arriving. "
                                 "Check the existing driver and Ethernet configuration.")
            model = profile["driver_model"]
            if model not in ("mid360", "mid360s"):
                raise ValueError("Invalid Livox model in settings.")
            launch = "msg_MID360s_launch.py" if model == "mid360s" else "msg_MID360_launch.py"
            return wait(start(["ros2", "launch", "livox_ros_driver2", launch], profile, driver=True))
        if not ready:
            raise ValueError("LiDAR and IMU data are required. Start the sensor driver first "
                             "('python3 scripts/onboard.py lidar' on AGX), then retry.")
        if any(n in observed["nodes"] for n in ("lightning_slam", "lightning_localization")):
            raise ValueError("A Lightning-LM application is already running in this ROS domain. Stop it first.")
        command = ["ros2", "run", "lightning", "run_slam_online" if args.command == "slam" else "run_loc_online",
                   "--config", profile["config"], "--rviz", "--console=" + args.console]
        if args.command == "slam" and args.record_bag:
            command += ["--record_bag"]
        if args.command == "localize":
            if args.global_init:
                command += ["--global_init"]
            command += ["--map_path", str(args.map.expanduser().resolve()), "--trajectory",
                        str(args.trajectory.expanduser().resolve()), "--", "--ros-args",
                        "-r", "/tf:=/lightning/tf", "-r", "/initialpose:=/lightning/initialpose"]
        if args.command == "localize":
            return wait(start(command, profile, cpus=profile.get("app_cpus")))
        session_file = args.settings.with_suffix(".session.json")
        session = dict(pid=os.getpid(), process_start=process_identity(os.getpid()),
                       imu_reference=observed.get("imu_reference"))
        write_atomic(session_file, json.dumps(session) + "\n")
        if not session["imu_reference"]:
            print("No stable startup gravity reference; saved maps will retain their original view frame.", flush=True)
        try:
            return wait(start(command, profile, cpus=profile.get("app_cpus")))
        finally:
            if mapping_session(args.settings) == session:
                session_file.unlink(missing_ok=True)



if __name__ == "__main__":
    # SSH disconnects and service stops must run the same child cleanup as Ctrl+C.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, yaml.YAMLError, subprocess.TimeoutExpired) as error:
        print("onboard: ERROR:", error, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
