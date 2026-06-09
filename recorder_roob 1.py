from pyorbbecsdk import *
import cv2
import numpy as np
import time
import csv
import os
from threading import Lock
from utils import frame_to_bgr_image

"""
Changes made to Rabias Original: pipeline includes also gyro and accel files
"""

ESC_KEY = 27
trial = 6# <-- change this number for each trial
console_lock = Lock()
# Create trial folder
data_folder = os.path.join(os.path.expanduser("~"), "Desktop", "RecordingsMariona", str(trial))
# data_folder = rf"C:\Users\Rabia\source\repos\PythonApplication1\Data\Mariona\V{trial}"

# Check if the folder already exists, if so the user needs to decide how to proceed
if os.path.exists(data_folder):
    user_input = input(f"The folder {data_folder} already exists. Do you want to overwrite it? (y/n): ").lower()
    if user_input != 'y':
        print("Exiting program...")
        exit()


os.makedirs(data_folder, exist_ok=True)

# CSV file paths
gyro_file = os.path.join(data_folder, "gyro_test.csv")
accel_file = os.path.join(data_folder, "accel_test.csv")
IMU_file = os.path.join(data_folder, "IMU_data.csv")

# CSV headers
gyro_columns = ['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
accel_columns = ['Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z']
IMU_columns = ['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z','Accel_X', 'Accel_Y', 'Accel_Z']


# Write header if file doesn't exist
def write_header(csv_file_path, column_names):
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerow(column_names)

write_header(gyro_file, gyro_columns)
write_header(accel_file, accel_columns)
write_header(IMU_file, IMU_columns)

# Stop flags
stop_gyro = False
stop_accel = False

def on_gyro_frame_callback(frame):
    if frame is None or stop_gyro:
        return
    with console_lock:
        gyro_frame = frame.as_gyro_frame()
        if gyro_frame is not None:
            ts = gyro_frame.get_timestamp()
            x, y, z = gyro_frame.get_x(), gyro_frame.get_y(), gyro_frame.get_z()
            with open(gyro_file, 'a', newline='') as csvfile:
                csv.writer(csvfile).writerow([ts, x, y, z])
            print(f"Gyro: ts={ts}, x={x}, y={y}, z={z}")

def on_accel_frame_callback(frame):
    if frame is None or stop_accel:
        return
    with console_lock:
        accel_frame = frame.as_accel_frame()
        if accel_frame is not None:
            ts = accel_frame.get_timestamp()
            x, y, z = accel_frame.get_x(), accel_frame.get_y(), accel_frame.get_z()
            with open(accel_file, 'a', newline='') as csvfile:
                csv.writer(csvfile).writerow([ts, x, y, z])
            print(f"Accel: ts={ts}, x={x}, y={y}, z={z}")

def data_sync(gyro_csv, accel_csv, imu_csv):

    accel_dict = {}
    with open(accel_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            accel_dict[row['Timestamp']] = row

    with open(imu_csv, 'a', newline='') as imu_f:
        writer = csv.writer(imu_f)

        with open(gyro_csv, 'r', newline='') as gyro_f:
            reader = csv.DictReader(gyro_f)
            for g_row in reader:
                ts = g_row['Timestamp']

                # Match timestamps
                if ts in accel_dict:
                    a_row = accel_dict[ts]
                    writer.writerow([
                        ts,
                        g_row['Gyro_X'], g_row['Gyro_Y'], g_row['Gyro_Z'],
                        a_row['Accel_X'], a_row['Accel_Y'], a_row['Accel_Z']
                    ])

def main():
    pipeline = Pipeline()
    config = Config()
    device = pipeline.get_device()
    sensor_list: SensorList = device.get_sensor_list()

    # Enable depth stream
    try:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
    except Exception as e:
        print("Depth stream error:", e)

    # Enable color stream
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(profile)

        # Enable color stream that matches depth resolution
        # color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # color_profile = None
        # for p in color_profiles:
        #     if p.get_width() == depth_profile.get_width() and p.get_height() == depth_profile.get_height():
        #         color_profile = p
        #         break
        # if color_profile is None:
        #     color_profile = color_profiles.get_default_video_stream_profile()
        # config.enable_stream(color_profile)
    except Exception as e:
        print("Color stream error:", e)

    # Enable IR stream
    try:
        ir_profiles = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
        profile = ir_profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
    except Exception as e:
        print("IR stream error:", e)

    # Start gyro sensor callback
    try:
        gyro_sensor = sensor_list.get_sensor_by_type(OBSensorType.GYRO_SENSOR)
        if gyro_sensor is not None:
            gyro_profiles = gyro_sensor.get_stream_profile_list()
            profile = gyro_profiles.get_stream_profile_by_index(0)
            gyro_sensor.start(profile, on_gyro_frame_callback)
    except Exception as e:
        print("Gyro sensor error:", e)

    # Start accel sensor callback
    try:
        accel_sensor = sensor_list.get_sensor_by_type(OBSensorType.ACCEL_SENSOR)
        if accel_sensor is not None:
            accel_profiles = accel_sensor.get_stream_profile_list()
            profile = accel_profiles.get_stream_profile_by_index(0)
            accel_sensor.start(profile, on_accel_frame_callback)
    except Exception as e:
        print("Accel sensor error:", e)

    # Start pipeline recording
    pipeline.start(config)
    pipeline.start_recording(os.path.join(data_folder, f"V{trial}.bag"))

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            master_ts = frames.get_timestamp() 
            print(master_ts)
            # Depth frame
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                width, height = depth_frame.get_width(), depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
                depth_data = depth_data.astype(np.float32) * scale
                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                cv2.imshow("Depth Viewer", depth_image)

                # # Save raw depth with master timestamp
                # master_ts = depth_frame.get_timestamp()
                # np.save(os.path.join(data_folder, f"depth_{master_ts}.npy"), depth_data)

            # # IR frame
            # ir_frame = frames.get_ir_frame()
            # if ir_frame is not None:
            #     width, height = ir_frame.get_width(), ir_frame.get_height()
            #     ir_data = np.frombuffer(ir_frame.get_data(), dtype=np.uint16).reshape((height, width))
            #     ir_image = cv2.normalize(ir_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #     #cv2.imshow("IR Viewer", ir_image)
            #     ir_frame = frames.get_ir_frame()
            #     master_ts = ir_frame.get_timestamp()
            #     np.save(os.path.join(data_folder, f"ir_{master_ts}.npy"), ir_data)
            
            # # Color Frame
            # color_frame = frames.get_color_frame()
            # if color_frame is not None:
            #     color_image = frame_to_bgr_image(color_frame)
            #     master_ts = color_frame.get_timestamp()
            #     np.save(os.path.join(data_folder, f"color_{master_ts}.npy"), color_image.astype(np.uint8))
            #     color_frame = frames.get_color_frame()

            key = cv2.waitKey(1)
            if key == ESC_KEY or key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")

    # Stop sensors and pipeline
    global stop_gyro, stop_accel
    stop_gyro = True
    stop_accel = True
    time.sleep(0.01)

    if gyro_sensor is not None:
        gyro_sensor.stop()
    if accel_sensor is not None:
        accel_sensor.stop()

    pipeline.stop_recording()
    pipeline.stop()
    cv2.destroyAllWindows()
        
    
    # Synchronize IMU data after recording stops
    #data_sync(gyro_file, accel_file, IMU_file)

    print(f"Trial {trial} saved in {data_folder}")

if __name__ == "__main__":
    main()
