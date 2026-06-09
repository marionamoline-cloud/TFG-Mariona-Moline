# -*- coding: utf-8 -*-

# ---------------------------------------------------
# SCRIPT FOR FAST FRAME EXTRACTION
# Input: .bag file obtained from pyorbbecsdk recorder.py (Only for older SDKs, v.1.8!!!)
# Output: Image frames from color, depth and infrared channel
# ---------------------------------------------------

import sys
import os
import cv2
import csv
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image


class BagProcessor:

    def __init__(self, base_dir, trial):

        self.trial = trial

        self.BASE_DIR = base_dir
        self.OUTPUT_DIR = os.path.join(base_dir, "Output_fast")

        self.output_color = os.path.join(self.OUTPUT_DIR, "COLOR_RAW")
        self.output_depth = os.path.join(self.OUTPUT_DIR, "DEPTH_RAW")
        self.output_ir = os.path.join(self.OUTPUT_DIR, "IR_RAW")
        self.camera_param_file = os.path.join(self.BASE_DIR, "camera_params.npz")

        for folder in [
            self.OUTPUT_DIR,
            self.output_color,
            self.output_depth,
            self.output_ir,
        ]:
            os.makedirs(folder, exist_ok=True)

        self.saved_frames = []
        self.fx = self.fy = self.cx = self.cy = None
        self.rot = None
        self.translation_vector = None

    def extract_frames(self):

        print("Starting data extraction...")

        bag_path = os.path.join(self.BASE_DIR, f"V{self.trial}.bag")
        pipeline = Pipeline(bag_path)
        playback = pipeline.get_playback()

        pipeline.start()

        # Camera parameters
        camera_param = pipeline.get_camera_param()
        self.fx = camera_param.depth_intrinsic.fx
        self.fy = camera_param.depth_intrinsic.fy
        self.cx = camera_param.depth_intrinsic.cx
        self.cy = camera_param.depth_intrinsic.cy
        self.rot = camera_param.transform.rot
        self.translation_vector = camera_param.transform.transform
        self.depth_dist = camera_param.depth_distortion
        self.rgb_dist   = camera_param.rgb_distortion

        np.savez(
            self.camera_param_file,

            rgb_intrinsics=np.array([
                camera_param.rgb_intrinsic.fx,
                camera_param.rgb_intrinsic.fy,
                camera_param.rgb_intrinsic.cx,
                camera_param.rgb_intrinsic.cy,
                camera_param.rgb_intrinsic.width,
                camera_param.rgb_intrinsic.height
            ]),

            depth_intrinsics=np.array([
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                camera_param.depth_intrinsic.width,
                camera_param.depth_intrinsic.height
            ]),

            rotation=np.array(camera_param.transform.rot),
            translation=np.array(camera_param.transform.transform),

            depth_dist_coeffs = np.array(
            [self.depth_dist.k1, self.depth_dist.k2,
             self.depth_dist.p1, self.depth_dist.p2,
             self.depth_dist.k3],
            dtype=np.float32),

            rgb_dist_coeffs = np.array(
            [self.rgb_dist.k1, self.rgb_dist.k2,
             self.rgb_dist.p1, self.rgb_dist.p2,
             self.rgb_dist.k3],
            dtype=np.float32)

                )   

        frame_index = 1
        no_frame_counter = 0
        max_no_frame = 10  # stop after 10 consecutive empty frames

        while True:
            frames = pipeline.wait_for_frames(1000)

            if frames is None:
                no_frame_counter += 1
                if no_frame_counter >= max_no_frame:
                    print("End of bag detected")
                    break
                continue
            else:
                no_frame_counter = 0

            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_ir_frame()
            color_frame = frames.get_color_frame() 

 

            if depth_frame is None or ir_frame is None or color_frame is None:

                continue

            # ----- DEPTH -----
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale


            # depth_path = os.path.join(self.output_depth, f"depth_{frame_index:04d}.npy")
            # np.save(depth_path, depth_data.astype(np.float32))
            master_ts = depth_frame.get_timestamp()
            np.save(os.path.join(self.output_depth, f"depth_{master_ts}.npy"), depth_data)

            # ----- COLOR -----
            color_image = frame_to_bgr_image(color_frame)
            master_ts_color = color_frame.get_timestamp()
            np.save(os.path.join(self.output_color, f"color_{master_ts_color}.npy"), color_image.astype(np.uint8))

            # color_path = os.path.join(self.output_color, f"color_{frame_index:04d}.npy")
            # np.save(color_path, color_image.astype(np.uint8))            
            # cv2.imwrite(os.path.join(self.output_color, f"color_{frame_index}.jpg"), color_image)

            # ----- IR -----
            ir_data = np.frombuffer(ir_frame.get_data(), dtype=np.uint16)

            master_ts_ir = ir_frame.get_timestamp()
            np.save(os.path.join(self.output_ir, f"ir_{master_ts_ir}.npy"), ir_data)
            
            # ir_path = os.path.join(self.output_ir, f"ir_{frame_index:04d}.npy")
            # np.save(ir_path, ir_data)

            #ir_image = ir_data.reshape((height, width))
            #cv2.imwrite(os.path.join(self.output_ir, f"ir_{frame_index}.png"), ir_image)

            self.saved_frames.append(frame_index)
            frame_index += 1

        pipeline.stop()
        print("Extraction complete.")
        print("Program finished.")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    trial = 6



    BASE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "RecordingsMariona", str(trial))

    processor = BagProcessor(BASE_DIR, trial)
    processor.extract_frames()

    sys.exit(0)
