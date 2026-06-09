# -*- coding: utf-8 -*-

import sys

import os

import cv2

import csv

import numpy as np



trial = "mariona1"

NUM_MARKERS = 6

ESC_KEY = 27



expected_radius_px = 8 # average marker radius in IR image

radius_tolerance =0.9

min_radius = expected_radius_px * (1 - radius_tolerance)

max_radius = expected_radius_px * (1 + radius_tolerance)

min_circularity = 0.6    # 1.0 = perfect circle


BASE_DIR = os.path.join(r"D:\Mariona\RecordingsMariona", str(trial))

OUTPUT_DIR = os.path.join(BASE_DIR, "Output_processed")

points3d_file = os.path.join(OUTPUT_DIR, "points_3d.csv")


DEPTH_FOLDER = os.path.join(BASE_DIR, "Output_fast", "DEPTH_RAW")

IR_FOLDER = os.path.join(BASE_DIR, "Output_fast", "IR_RAW")

COLOR_FOLDER = os.path.join(BASE_DIR, "Output_fast", "COLOR_RAW")



# Output folders

OUTPUT_COLOR = os.path.join(OUTPUT_DIR, "COLOR")

OUTPUT_DEPTH = os.path.join(OUTPUT_DIR, "DEPTH")

OUTPUT_IR = os.path.join(OUTPUT_DIR, "IR")

OUTPUT_OVERLAY = os.path.join(OUTPUT_DIR, "OUTPUT")



for folder in [OUTPUT_DIR, OUTPUT_DEPTH, OUTPUT_COLOR, OUTPUT_IR, OUTPUT_OVERLAY]:

    os.makedirs(folder, exist_ok=True)



# CSV paths

centroids_file = os.path.join(OUTPUT_DIR, "centers_2d.csv")

points3d_file = os.path.join(OUTPUT_DIR, "points_3d.csv")

transform3d_file = os.path.join(OUTPUT_DIR, "transform_3d.csv")



# CSV headers

with open(centroids_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([f"pt{i}_{c}" for i in range(1, NUM_MARKERS+1) for c in ("x","y")])

with open(points3d_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([f"pt{i}_{c}" for i in range(1, NUM_MARKERS+1) for c in ("x","y","z")])

with open(transform3d_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([f"pt{i}_{c}" for i in range(1, NUM_MARKERS+1) for c in ("x","y","z")])



# ------------------------

# HELPER FUNCTIONS

# ------------------------



def point2d_to_point3d(fx, fy, cx, cy, depth_data, overlay_image, point, radius):

    x, y = point

    radius += 3.0 # expected radius is 5 pixels, so better to decrease, in case center of circle is not perfectly detected

    depths = []

    height, width = depth_data.shape



    for i in range(round(x - radius), round(x + radius)+1):

        for j in range(round(y - radius), round(y + radius)+1):

            if 0 <= i < width and 0 <= j < height:

                if (i-x)**2 + (j-y)**2 <= radius**2 and depth_data[j,i] > 0:

                    depths.append(depth_data[j,i])

                    #overlay_image[j,i] = (0,255,0)



    if 0 <= int(round(x)) < width and 0 <= int(round(y)) < height:

        cv2.circle(overlay_image, (int(round(x)), int(round(y))), int(round(radius)), (0, 255, 0), 2)  # thickness=1



    if len(depths) == 0:

        return np.nan, np.nan, np.nan, overlay_image



    source_depth_mm = np.mean(depths)

    x3d = ((x - cx) / fx) * source_depth_mm

    y3d = ((y - cy) / fy) * source_depth_mm

    z3d = source_depth_mm



    return x3d, y3d, z3d, overlay_image





def coordinate_transformation(point_3d, rot, transform):

    R = np.array(rot)

    T = np.array(transform)

    M = np.eye(4)

    M[:3,:3] = R

    M[:3,3] = T

    pt_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1])

    transformed = np.dot(M, pt_h)[:3]

    return transformed



# ------------------------

# MAIN PROCESSING

# ------------------------



def main():

    # Load all frames

    depth_files = sorted([f for f in os.listdir(DEPTH_FOLDER) if f.endswith(".npy")])

    ir_files = sorted([f for f in os.listdir(IR_FOLDER) if f.endswith(".npy")])

    color_files = sorted([f for f in os.listdir(COLOR_FOLDER) if f.endswith((".npy"))])

    print("Depth:", len(depth_files))

    print("IR:", len(ir_files))

    print("Color:", len(color_files))





    # Get camera intrinsics 

    camera_param = np.load(os.path.join(BASE_DIR, "camera_params.npz"))

    rgb = camera_param['rgb_intrinsics']

    depth = camera_param['depth_intrinsics']

    rot = camera_param['rotation']

    translation_vector = camera_param['translation']



    fx_rgb, fy_rgb, cx_rgb, cy_rgb, w_rgb, h_rgb = rgb

    fx_depth, fy_depth, cx_depth, cy_depth, w_depth, h_depth = depth

    

    



    prev_markers_2d = None





    for frame_index, (dfile, ifile, cfile) in enumerate(zip(depth_files, ir_files, color_files), 1):

        # Load frames

        depth_data = np.load(os.path.join(DEPTH_FOLDER, dfile))

        ir_data = np.load(os.path.join(IR_FOLDER, ifile))

        color_image = np.load(os.path.join(COLOR_FOLDER, cfile))



        height, width = depth_data.shape



        # Prepare depth image for overlay

        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

        #cv2.imshow("Depth Image", depth_image)



        # Process IR: threshold markers

        ir_image = ir_data.reshape((height, width))

        ir_image = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)

        ir_image = ir_image.astype(np.uint8)



        # Threshold for bright markers (adjust threshold value if needed)

        _, ir_binary = cv2.threshold(ir_image, 240, 255, cv2.THRESH_BINARY)  

        #cv2.imshow("IR Binary", ir_binary)



        analysis = cv2.connectedComponentsWithStats(ir_binary, 8, cv2.CV_32S)

        totalLabels, label_ids, stats, centroids = analysis



        contours, _ = cv2.findContours(ir_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_2d = []

        detected_3d = []

        overlay_image = depth_image.copy()



        for contour in contours:

            area = cv2.contourArea(contour)

            perimeter = cv2.arcLength(contour, True)

            if area <= 0 or perimeter==0: continue

            radius = np.sqrt(area / np.pi)

            print(radius)

            if not (min_radius <= radius <= max_radius): continue

            circularity = 4*np.pi*area/(perimeter**2)

            if circularity < min_circularity: continue

            (center, _) = cv2.minEnclosingCircle(contour)

            cx2d, cy2d = center



            x3d, y3d, z3d, overlay_image = point2d_to_point3d(fx_depth, fy_depth, cx_depth, cy_depth, depth_data, overlay_image, center, radius)

            detected_2d.append((cx2d, cy2d))

            detected_3d.append((x3d, y3d, z3d))



        #marker identification using nearest neighbors

        frame_2d = [None] * NUM_MARKERS

        frame_3d = [None] * NUM_MARKERS

        used = np.zeros(NUM_MARKERS, dtype=bool)



        if prev_markers_2d is None:

            for i in range(min(NUM_MARKERS, len(detected_2d))):

               frame_2d[i] = detected_2d[i]

               frame_3d[i] = detected_3d[i]

        else:

            prev = np.array(prev_markers_2d)

            used = np.zeros(NUM_MARKERS, dtype=bool)



            for p2d, p3d in zip(detected_2d, detected_3d):

               d = np.linalg.norm(prev - np.array(p2d), axis=1)

               d[used] = np.inf



               idx = np.argmin(d)

               frame_2d[idx] = p2d

               frame_3d[idx] = p3d

               used[idx] = True

        

 







        prev_markers_2d = [frame_2d[i] if frame_2d[i] is not None else prev_markers_2d[i] for i in range(NUM_MARKERS)]

        

        # Save overlays

        # overlay_image = depth_image.copy()

        # for cx2d, cy2d in frame_2d:

        #     if cx2d is not None:

        #         cv2.circle(overlay_image, (int(cx2d), int(cy2d)), 5, (0, 255, 0), 2)  # radius 5, thickness 2



        cv2.imwrite(os.path.join(OUTPUT_DEPTH, f"depth_{frame_index}.jpg"), depth_image)

        cv2.imwrite(os.path.join(OUTPUT_COLOR, f"color_{frame_index}.jpg"), color_image)

        cv2.imwrite(os.path.join(OUTPUT_IR, f"ir_{frame_index}.jpg"), ir_binary)

        cv2.imwrite(os.path.join(OUTPUT_OVERLAY, f"overlay_{frame_index}.jpg"), overlay_image)



        # Write CSVs

        row2d = []

        for m in frame_2d:

            if m is None:

                row2d.extend([np.nan, np.nan])

            else:

                row2d.extend(m)

        row3d = []

        frame_transformed = []



        for m in frame_3d:

            if m is None:

                row3d.extend([np.nan, np.nan, np.nan])

                frame_transformed.extend([np.nan, np.nan, np.nan])

            else:

                row3d.extend(m)

                tp = coordinate_transformation(m, rot, translation_vector)

                frame_transformed.extend(tp)



        with open(centroids_file, "a", newline="") as f:

            csv.writer(f).writerow(row2d)



        with open(points3d_file, "a", newline="") as f:

            csv.writer(f).writerow(row3d)



        with open(transform3d_file , "a", newline="") as f:

             csv.writer(f).writerow(frame_transformed)





if __name__ == "__main__":

    main()



### Loading of npy data ###


# import os
# import cv2
# import numpy as np

# BASE_DIR = r"C:\Users\Rabia\source\repos\PythonApplication1\Data\V2\Output_fast"
# COLOR_DIR = os.path.join(BASE_DIR, "COLOR_RAW")
# DEPTH_DIR = os.path.join(BASE_DIR, "DEPTH_RAW")
# IR_DIR = os.path.join(BASE_DIR, "IR_RAW")

# frame_index = 100  # pick a frame to debug

# # --- Load Color ---
# color_files = sorted(os.listdir(COLOR_DIR))
# color_path = os.path.join(COLOR_DIR, color_files[frame_index-1])
# color_image = np.load(color_path)
# cv2.imshow("Color Image", color_image)

# # --- Load Depth ---
# depth_files = sorted(os.listdir(DEPTH_DIR))
# depth_path = os.path.join(DEPTH_DIR, depth_files[frame_index-1])
# depth_data = np.load(depth_path)  # depth saved as float32 in mm
# print(depth_data.shape)
# print(f"Depth min/max: {np.min(depth_data):.2f} / {np.max(depth_data):.2f} mm")

# # Normalize depth to 0-255 for visualization
# depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
# depth_vis = depth_vis.astype(np.uint8)
# depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
# cv2.imshow("Depth Image", depth_vis_color)

# # --- Load IR ---
# ir_files = sorted(os.listdir(IR_DIR))
# ir_path = os.path.join(IR_DIR, ir_files[frame_index-1])
# ir_data = np.load(ir_path)  # if saved as npy
# ir_image = ir_data.reshape((576, 640))

# print(f"IR min/max: {np.min(ir_data)} / {np.max(ir_data)}")
# print(ir_image.shape)

# # Normalize IR for visualization
# ir_vis = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
# ir_vis = ir_vis.astype(np.uint8)
# ir_image = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)
# ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("IR Image", ir_vis)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
