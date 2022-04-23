#!/usr/bin/env python3
from datetime import timedelta, datetime
from os import error
from time import sleep
from numpy import mean
from requests import get, post
import re
import json
import logging
from typing import Dict

######### META DATA #################
# For data collection organizational purposes
USER_ID = "foonietunes"  # e.g. Discord handle
PRINTER_MODEL = "voron_v2_300"  # e.g. 'voron_v2_350'
HOME_TYPE = "nozzle_pin"  # e.g. 'nozzle_pin', 'microswitch_probe', etc.
PROBE_TYPE = "klicky"  # e.g. 'klicky', 'omron', 'bltouch', etc.
X_RAILS = "1x_mgn12_front"  # e.g. '1x_mgn12_front', '2x_mgn9'
BACKERS = "none"  # e.g. 'steel_x_y', 'Ti_x-steel_y', 'mgn9_y'
NOTES = ""  # anything note-worthy about this particular run,
#     no "=" characters
#####################################

######### CONFIGURATION #############
BASE_URL = "http://127.0.0.1:7125"  # printer URL (e.g. http://192.168.1.15)
# leave default if running locally
BED_TEMPERATURE = 110  # bed temperature for measurements
HE_TEMPERATURE = 100  # extruder temperature for measurements
STABLE_TIME = 0  # minutes between temperature comparison to stop the measurements
SOAK_TIME = 5  # minutes to wait for bed to heatsoak after reaching temp
MEASURE_GCODE = (
    "G28 Z"  # G-code called on repeated measurements, single line/macro only
)
QGL_CMD = (
    "QUAD_GANTRY_LEVEL"  # command for QGL; e.g. "QUAD_GANTRY_LEVEL" or None if no QGL.
)
MESH_CMD = "BED_MESH_CALIBRATE SAMPLES=1 PROBE_COUNT=11,11 RELATIVE_REFERENCE_INDEX=120"

# Full config section name of the frame temperature sensor
FRAME_SENSOR = "temperature_sensor frame_front"
# chamber thermistor config name. Change to match your own, or "" if none
# will also work with temperature_fan configs
CHAMBER_SENSOR = "temperature_sensor chamber"
# Extra temperature sensors to collect. Use same format as above but seperate
# quoted names with commas (if more than one).
EXTRA_SENSORS = {}
MONITOR_SENSOR = "frame_temp"

#####################################


MCU_Z_POS_RE = re.compile(r"(?P<mcu_z>(?<=stepper_z:)-*[0-9.]+)")

date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_FILENAME = "meshes_%s.json" % (date_str)
start_time = datetime.now() + timedelta(days=1)
index = 0
BASE_URL = BASE_URL.strip("/")  # remove any errant "/" from the address
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def gather_metadata():
    resp = get(BASE_URL + "/printer/objects/query?configfile").json()
    config = resp["result"]["status"]["configfile"]["settings"]

    # Gather Z axis information
    config_z = config["stepper_z"]
    if "rotation_distance" in config_z.keys():
        rot_dist = config_z["rotation_distance"]
        steps_per = config_z["full_steps_per_rotation"]
        micro = config_z["microsteps"]
        if config_z["gear_ratio"]:
            gear_ratio_conf = config_z["gear_ratio"]
            gear_ratio = float(gear_ratio_conf[0][0])
            for reduction in gear_ratio_conf[1:]:
                gear_ratio = gear_ratio / float(reduction)
        else:
            gear_ratio = 1.0
        step_distance = (rot_dist / (micro * steps_per)) / gear_ratio
    elif "step_distance" in config_z.keys():
        step_distance = config_z["step_distance"]
    else:
        step_distance = "NA"
    max_z = config_z["position_max"]
    if "second_homing_speed" in config_z.keys():
        homing_speed = config_z["second_homing_speed"]
    else:
        homing_speed = config_z["homing_speed"]

    # Organize
    meta = {
        "user": {
            "id": USER_ID,
            "printer": PRINTER_MODEL,
            "home_type": HOME_TYPE,
            "probe_type": PROBE_TYPE,
            "x_rails": X_RAILS,
            "backers": BACKERS,
            "notes": NOTES,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        },
        "script": {
            "data_structure": 3,
        },
        "z_axis": {
            "step_dist": step_distance,
            "max_z": max_z,
            "homing_speed": homing_speed,
        },
    }
    return meta


def write_metadata(meta):
    with open(DATA_FILENAME, "w") as dataout:
        dataout.write("### METADATA ###\n")
        for section in meta.keys():
            print(section)
            dataout.write("## %s ##\n" % section.upper())
            for item in meta[section]:
                dataout.write("# %s=%s\n" % (item, meta[section][item]))
        dataout.write("### METADATA END ###\n")


def query_axis_bounds(axis):
    resp = get(BASE_URL + "/printer/objects/query?configfile").json()
    config = resp["result"]["status"]["configfile"]["settings"]

    stepper = "stepper_%s" % axis

    axis_min = config[stepper]["position_min"]
    axis_max = config[stepper]["position_max"]

    return (axis_min, axis_max)


def query_xy_middle():
    resp = get(BASE_URL + "/printer/objects/query?configfile").json()
    config = resp["result"]["status"]["configfile"]["settings"]

    x_min = config["stepper_x"]["position_min"]
    x_max = config["stepper_x"]["position_max"]
    y_min = config["stepper_y"]["position_min"]
    y_max = config["stepper_y"]["position_max"]

    x_mid = x_max - (x_max - x_min) / 2
    y_mid = y_max - (y_max - y_min) / 2

    return [x_mid, y_mid]


def send_gcode_nowait(cmd=""):
    url = BASE_URL + "/printer/gcode/script?script=%s" % cmd
    post(url)
    return True


def send_gcode(cmd="", retries=1):
    url = BASE_URL + "/printer/gcode/script?script=%s" % cmd
    resp = post(url)
    success = None
    for i in range(retries):
        try:
            success = "ok" in resp.json()["result"]
        except KeyError:
            print("G-code command '%s', failed. Retry %i/%i" % (cmd, i + 1, retries))
        else:
            return True
    return False


def park_head_center():
    xy_coords = query_xy_middle()
    send_gcode_nowait("G1 Z10 F300")

    park_cmd = "G1 X%.1f Y%.1f F18000" % (xy_coords[0], xy_coords[1])
    send_gcode_nowait(park_cmd)


def park_head_high():
    xmin, xmax = query_axis_bounds("x")
    ymin, ymax = query_axis_bounds("y")
    zmin, zmax = query_axis_bounds("z")

    xpark = xmax
    ypark = ymax
    zpark = zmax * 0.8

    park_cmd = "G1 X%.1f Y%.1f Z%.1f F1000" % (xpark, ypark, zpark)
    send_gcode_nowait(park_cmd)


def set_bedtemp(t=0):
    temp_set = False
    cmd = "SET_HEATER_TEMPERATURE HEATER=heater_bed TARGET=%.1f" % t
    temp_set = send_gcode(cmd, retries=3)
    logging.info(cmd)
    if not temp_set:
        raise RuntimeError("Bed temp could not be set.")


def set_hetemp(t=0):
    temp_set = False
    cmd = "SET_HEATER_TEMPERATURE HEATER=extruder TARGET=%.1f" % t
    temp_set = send_gcode(cmd, retries=3)
    logging.info(cmd)
    if not temp_set:
        raise RuntimeError("HE temp could not be set.")


def gantry_leveled():
    url = BASE_URL + "/printer/objects/query?quad_gantry_level"
    resp = get(url).json()["result"]
    return resp["status"]["quad_gantry_level"]["applied"]


def qgl(retries=30):
    if not QGL_CMD:
        logging.info("No QGL; skipping.")
        return True
    if gantry_leveled():
        logging.info("Gantry already level. ")
        return True
    if not gantry_leveled():
        logging.info("Leveling gantry...")
        send_gcode_nowait(QGL_CMD)
        for attempt in range(retries):
            if gantry_leveled():
                return True
            else:
                sleep(10)

    raise RuntimeError("Could not level gantry")


def clear_bed_mesh():
    mesh_cleared = False
    cmd = "BED_MESH_CLEAR"
    mesh_cleared = send_gcode(cmd, retries=3)
    if not mesh_cleared:
        raise RuntimeError("Could not clear mesh.")


def take_bed_mesh():
    mesh_received = False
    cmd = MESH_CMD

    # print("Taking bed mesh measurement...", end="", flush=True)
    send_gcode_nowait(cmd)

    mesh = query_bed_mesh()
    return mesh


def query_bed_mesh(retries=60):
    url = BASE_URL + "/printer/objects/query?bed_mesh"
    mesh_received = False
    for attempt in range(retries):
        resp = get(url).json()["result"]
        mesh = resp["status"]["bed_mesh"]
        if mesh["mesh_matrix"] != [[]]:
            mesh_received = True
            return mesh
        else:
            sleep(10)
    if not mesh_received:
        raise RuntimeError("Could not retrieve mesh")


def query_temp_sensors():
    extra_t_str = ""
    if CHAMBER_SENSOR:
        extra_t_str += "&%s" % CHAMBER_SENSOR
    if FRAME_SENSOR:
        extra_t_str += "&%s" % FRAME_SENSOR
    if EXTRA_SENSORS:
        extra_t_str += "&%s" % "&".join(EXTRA_SENSORS.values())

    base_t_str = "extruder&heater_bed"
    url = BASE_URL + "/printer/objects/query?{0}{1}".format(base_t_str, extra_t_str)
    resp = get(url).json()["result"]["status"]
    try:
        chamber_current = resp[CHAMBER_SENSOR]["temperature"]
    except KeyError:
        chamber_current = -180.0
    try:
        frame_current = resp[FRAME_SENSOR]["temperature"]
    except KeyError:
        frame_current = -180.0

    extra_temps = {}
    if EXTRA_SENSORS:
        for sensor in EXTRA_SENSORS:
            try:
                extra_temps[sensor] = resp[EXTRA_SENSORS[sensor]]["temperature"]
            except KeyError:
                extra_temps[sensor] = -180.0

    bed_current = resp["heater_bed"]["temperature"]
    bed_target = resp["heater_bed"]["target"]
    he_current = resp["extruder"]["temperature"]
    he_target = resp["extruder"]["target"]
    return {
        "frame_temp": frame_current,
        "chamber_temp": chamber_current,
        "bed_temp": bed_current,
        "bed_target": bed_target,
        "he_temp": he_current,
        "he_target": he_target,
        **extra_temps,
    }


def combine_temp_reading(temp1: Dict, temp2: Dict) -> Dict:
    temp = {}
    for k in temp1.keys():
        temp.update({k: (temp1[k], temp2[k])})
    return temp


def get_cached_gcode(n=1):
    url = BASE_URL + "/server/gcode_store?count=%i" % n
    resp = get(url).json()["result"]["gcode_store"]
    return resp


def query_mcu_z_pos():
    send_gcode(cmd="get_position")
    gcode_cache = get_cached_gcode(n=1)
    for msg in gcode_cache:
        pos_matches = list(MCU_Z_POS_RE.finditer(msg["message"]))
        if len(pos_matches) > 1:
            return int(pos_matches[0].group())


def wait_for_bedtemp(soak_time=5):
    while 1:
        temps = query_temp_sensors()
        if temps["bed_temp"] >= BED_TEMPERATURE - 0.5:
            logging.info(
                f"Bed temp reached target. Heat soaking for {soak_time} minutes..."
            )
            sleep(soak_time * 60)
            break


def pretty_temp(temp):
    temp = sorted(temp.items())
    return " ".join([f"{k}:{v}" for k, v in temp])


def collect_datapoint(idx):
    global temps, measurements

    if not send_gcode(MEASURE_GCODE):
        set_bedtemp()
        set_hetemp()
        err = "MEASURE_GCODE (%s) failed. Stopping." % MEASURE_GCODE
        raise RuntimeError(err)
    now = datetime.now()
    timestamp = now.strftime("%Y/%m/%d-%H:%M:%S")
    start_temp = query_temp_sensors()
    mesh = take_bed_mesh()
    end_temp = query_temp_sensors()
    logging.info(pretty_temp(start_temp))
    temps.append(
        {
            "time": now,
            MONITOR_SENSOR: mean(
                (start_temp[MONITOR_SENSOR], end_temp[MONITOR_SENSOR])
            ),
        }
    )
    measurements.update(
        {
            timestamp: {
                "idx": idx,
                "mesh": mesh,
                "pos": query_mcu_z_pos(),
                **combine_temp_reading(start_temp, end_temp),
            }
        }
    )


def temp_is_changing():
    global temps

    temps_before = [
        x
        for x in temps
        if (datetime.now() - x["time"]) >= timedelta(minutes=STABLE_TIME)
    ]

    if len(temps_before) >= 5:
        temp_prev = mean([x[MONITOR_SENSOR] for x in temps_before][-5:])
        temp_curr = mean([x[MONITOR_SENSOR] for x in temps][-5:])
        diff = temp_curr - temp_prev
        logging.info(f"cur: {temp_curr}, prev: {temp_prev}, diff:{diff}")

        if round(diff, 2) <= 0:
            logging.info(
                f"Mean of the last 5 {MONITOR_SENSOR} readings did not increase more than {STABLE_TIME} minutes"
            )
            return 0

    return 1


def exit():
    global measurements, temps, metadata, pre_data
    output = {
        "metadata": metadata,
        "pre_mesh": pre_data,
        "measurements": measurements,
        "temp_data": temps,
    }

    with open(DATA_FILENAME, "w") as out_file:
        json.dump(output, out_file, indent=4, sort_keys=True, default=str)
    logging.info(f"results written to {DATA_FILENAME}")

    set_bedtemp()
    set_hetemp()
    send_gcode("SET_FRAME_COMP enable=1")
    logging.info("Turning off heaters and re-enabling frame compensation")


def main():
    global measurements, temps, metadata, pre_data
    temps = []
    measurements = {}

    metadata = gather_metadata()
    logging.debug("Started. Homing...")
    if not send_gcode("G28"):
        raise RuntimeError("Failed to home. Aborted.")

    clear_bed_mesh()
    qgl()

    if not send_gcode("G28"):
        raise RuntimeError("Failed to home. Aborted.")

    send_gcode("SET_FRAME_COMP enable=0")
    logging.info("Disabling frame compensation")

    # Take preheat mesh
    take_bed_mesh()
    pre_time = datetime.now()
    pre_mesh = query_bed_mesh()
    pre_temps = query_temp_sensors()
    pre_data = {"time": pre_time, "temps": pre_temps, "mesh": pre_mesh}

    # Turning on heaters
    set_bedtemp(BED_TEMPERATURE)
    set_hetemp(HE_TEMPERATURE)
    park_head_high()
    wait_for_bedtemp(soak_time=SOAK_TIME)

    # Mesh until temperature no longer increases above threshold
    logging.info("Measurements started")
    idx = 0
    while temp_is_changing():
        collect_datapoint(idx)
        idx += 1
    logging.info("Measurements complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Aborted by user!")
    except Exception as e:
        logging.error("Other error", exc_info=True)

    exit()
