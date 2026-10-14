"""Saved robot configuration and child-process lifecycle."""

import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
RVIZ = str(ROOT / "install/lightning/lib/lightning/run_rviz")


def process_identity(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]



def mapping_session(settings):
    path = settings.with_suffix(".session.json")
    try:
        session = json.loads(path.read_text())
        if process_identity(session["pid"]) == session["process_start"]:
            return session
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return None



def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)



def cpu_list(value):
    if value is None:
        return None
    if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", value):
        raise ValueError("CPU lists must look like 7 or 4-7.")
    for part in value.split(","):
        bounds = [int(v) for v in part.split("-")]
        first, last = bounds[0], bounds[-1]
        if first > last or last >= (os.cpu_count() or 1):
            raise ValueError("CPU list contains an unavailable CPU: " + value)
    return value



def find_driver_workspace(model):
    launch = "msg_MID360s_launch.py" if model == "mid360s" else "msg_MID360_launch.py"
    candidates = set()
    for pattern in ("*/install/setup.bash", "*/*/install/setup.bash"):
        for setup in Path.home().glob(pattern):
            if any(setup.parent.rglob(launch)):
                candidates.add(setup.parent.parent.resolve())
        if candidates:
            break
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in sorted(candidates)) or "none"
        raise ValueError("Expected one built full Livox driver workspace under your home directory; "
                         "found " + found + ". Configure with --driver-workspace PATH.")
    return str(candidates.pop())



def configure(args):
    previous = json.loads(args.settings.read_text()) if args.settings.exists() else {}
    if previous and previous.get("platform") != args.platform:
        raise ValueError("This checkout is configured for another robot. Use a separate checkout/settings file.")
    domain = args.domain if args.domain is not None else previous.get(
        "domain", 42 if args.platform == "agx" else int(os.environ.get("ROS_DOMAIN_ID", "0")))
    if not 0 <= domain <= 232:
        raise ValueError("ROS domain must be between 0 and 232.")
    driver = args.driver_workspace or previous.get("driver_workspace")
    model = args.driver_model or previous.get("driver_model", "mid360s")
    if args.platform == "agx":
        if not driver:
            driver = find_driver_workspace(model)
        if not driver or not (Path(driver).expanduser() / "install/setup.bash").is_file():
            raise ValueError("Set --driver-workspace to the built full Livox driver workspace.")
        driver = str(Path(driver).expanduser().resolve())
        launch = "msg_MID360s_launch.py" if model == "mid360s" else "msg_MID360_launch.py"
        if not any((Path(driver) / "install").rglob(launch)):
            raise ValueError("The full driver workspace must install " + launch + "; the Lightning-LM "
                             "message-only package cannot start the LiDAR.")
    config_path = args.settings.with_suffix(".yaml")
    config_text = None
    if args.config or not config_path.exists():
        sensor = args.config or ROOT / "config" / ("mid360.yaml" if args.platform == "agx" else "m20_pro.yaml")
        config = yaml.safe_load(Path(sensor).expanduser().read_text())
        config["system"]["with_ui"] = False
        if args.config is None:
            # Reduce onboard LiDAR correction work; keep recorded-data presets unchanged.
            config["fasterlio"]["skip_lidar_num"] = 2
        if args.platform == "agx" and args.config is None:
            # Livox manual: IMU origin in LiDAR coordinates is the inverse translation.
            config["fasterlio"]["extrinsic_T"] = [-0.011, -0.02329, 0.04412]
            config["fasterlio"]["extrinsic_R"] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        config_text = yaml.safe_dump(config, sort_keys=False)
    profile = dict(version=1, platform=args.platform, config=str(config_path), domain=domain,
                   driver_workspace=driver,
                   driver_model=model,
                   app_cpus=cpu_list(args.app_cpus if args.app_cpus is not None else previous.get(
                       "app_cpus", "7" if args.platform == "m20" else None)),
                   rviz_cpus=cpu_list(args.rviz_cpus if args.rviz_cpus is not None else previous.get(
                       "rviz_cpus", "6" if args.platform == "m20" else None)))
    if config_text is not None:
        write_atomic(config_path, config_text)
    write_atomic(args.settings, json.dumps(profile, indent=2) + "\n")
    print("Saved settings:", args.settings)
    print("Runtime configuration:", config_path)
    print("Later commands load both automatically; no shell exports are needed.")



def load_profile(path):
    if not path.is_file():
        raise ValueError("Run 'python3 scripts/onboard.py configure agx' "
                         "or 'configure m20' once first.")
    try:
        profile = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in onboard settings {path}: {error.msg}.") from error
    if (not isinstance(profile, dict) or profile.get("version") != 1
            or profile.get("platform") not in ("agx", "m20")):
        raise ValueError("Unsupported onboard settings: " + str(path))
    if not isinstance(profile.get("domain"), int) or not 0 <= profile["domain"] <= 232:
        raise ValueError("Invalid ROS domain in onboard settings.")
    if not Path(profile["config"]).is_file():
        raise ValueError("Runtime configuration is missing: " + profile["config"])
    for key in ("app_cpus", "rviz_cpus"):
        cpu_list(profile.get(key))
    return profile



def environment(profile, driver=False, viewer=False):
    # Build each child environment independently. In particular, the Livox driver
    # must not inherit this checkout's message-only package through an old overlay.
    env = dict(os.environ)
    for key in ("AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "COLCON_PREFIX_PATH", "PYTHONPATH",
                "LD_LIBRARY_PATH", "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION",
                "RMW_IMPLEMENTATION", "FASTRTPS_DEFAULT_PROFILES_FILE"):
        env.pop(key, None)
    exports = dict(ROS_DOMAIN_ID=str(profile["domain"]),
                   ROS_LOCALHOST_ONLY="1" if profile["platform"] == "agx" else "0",
                   RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
                   FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT / "config/fastdds.xml"))
    prelude = ""
    if profile["platform"] == "m20":
        if os.geteuid() != 0:
            raise ValueError("M20 applications need the root shell described in README section 6.1.")
        prelude += "source /opt/robot/scripts/setup_ros2.sh || exit $?\n"
    prelude += "export " + " ".join(k + "=" + shlex.quote(v) for k,v in exports.items()) + "\n"
    if driver:
        distro_file = ROOT / "install/lightning/share/lightning/ros_distro"
        distro = distro_file.read_text().strip()
        if distro not in ("foxy", "humble"):
            raise ValueError("Rebuild Lightning-LM with ROS Foxy or Humble first.")
        sources = [Path("/opt/ros") / distro / "setup.bash",
                   Path(profile["driver_workspace"]) / "install/setup.bash"]
    else:
        sources = [ROOT / "scripts/setup.bash"]
    for source in sources:
        prelude += "source " + shlex.quote(str(source)) + " || exit $?\n"
    if viewer:
        if not env.get("DISPLAY"):
            raise ValueError("No X11 display. Open a new laptop desktop terminal and connect with "
                             "ssh -Y -C (for AGX, use -J as in README section 6.2).")
        exports.update(__GLX_VENDOR_LIBRARY_NAME="mesa", LIBGL_ALWAYS_SOFTWARE="1",
                       LP_NUM_THREADS="2", QT_X11_NO_MITSHM="1")
        prelude += "export " + " ".join(k + "=" + shlex.quote(v) for k,v in exports.items()) + "\n"
    # Firmware/ROS setup can print notices. Keep those on stderr so a child's
    # stdout (in particular the sensor probe's JSON) remains machine-readable.
    # A brace group preserves the sourced environment in this shell.
    return env, "{\n" + prelude + '} >&2\nexec "$@"'



def start(command, profile, driver=False, viewer=False, cpus=None, **kwargs):
    env, prelude = environment(profile, driver, viewer)
    if cpus:
        command = ["taskset", "-c", cpu_list(cpus)] + command
    return subprocess.Popen(["bash", "--noprofile", "--norc", "-c", prelude,
                             "lightning-onboard"] + command, cwd=str(ROOT), env=env,
                            start_new_session=True, **kwargs)



def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if process.returncode is not None and process.returncode < 0 and process.returncode not in (-2, -15):
        print("onboard: child", process.pid, "terminated by", signal.Signals(-process.returncode).name,
              file=sys.stderr)



def wait(process):
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        stop(process)

