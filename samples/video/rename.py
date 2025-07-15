# import os
# import re

# # Set the path to the directory you want to rename files in
# folder_path = r'samples\video\MCFD'  # Change as needed

# # Regex to match files like "filename (1).ext"
# pattern = re.compile(r'^(.*) \((\d+)\)(\.[^.]+)$')

# # Collect and sort files for consistent camera grouping
# files = sorted(os.listdir(folder_path))

# for index, filename in enumerate(files):
#     match = pattern.match(filename)
#     if match:

#         base_name, number, extension = match.groups()
#         number = int(number)
#         new_number = f"{number:03d}"

#         # Calculate camera number (groups of 24 files)
#         camera_number = ((number - 1) // 24) + 1

#         # Build new filename
#         new_filename = f"{base_name}_{new_number}_c{camera_number}{extension}"

#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)

#         os.rename(old_path, new_path)
#         print(f'Renamed: {filename} -> {new_filename}')

# import cv2
# import os

# mp4_path = r'samples\video\MCFD\sample_003_c1.mp4'
# avi_path =  r'samples\video\MCFD\sample_003_c1.avi'

# print(f"Converting {mp4_path} to AVI...")

# # Open input video
# cap = cv2.VideoCapture(mp4_path)

# # Get video properties
# fps = cap.get(cv2.CAP_PROP_FPS)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Common for AVI

# # Create output video writer
# out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     out.write(frame)

# cap.release()
# out.release()
# print(f"Saved: {avi_path}")



# import os
# import cv2

# input_folder = r'samples\video\MCFD'
# output_folder = r'samples\video\MCFD\converted'

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
# correct_fps = 30  # Set the desired FPS

# # Walk through files
# for file_name in os.listdir(input_folder):
#     if not file_name.endswith(".avi"):
#         continue

#     input_path = os.path.join(input_folder, file_name)
#     output_path = os.path.join(output_folder, file_name)

#     # Target FPS (real-world fps you want)
#     target_fps = 30

#     # Open input video
#     cap = cv2.VideoCapture(input_path)

#     if not cap.isOpened():
#         print("Error: Cannot open video.")
#         exit()

#     # Get frame size
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Define codec and create VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # For .avi
#     out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

#     # Read and write all frames
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
#         frame_count += 1

#     cap.release()
#     out.release()

#     print(f"✅ Done. {frame_count} frames written at {target_fps} FPS → {output_path}")           

import os
import re

# Path to the folder containing files
folder_path = r'samples\video\Le2i'

# Regex to match filenames like "sample01_sh01_cam1.avi"
pattern = re.compile(r'^(sample)(\d{2})_(sh\d{2})_(cam\d+)\.avi$', re.IGNORECASE)

for filename in os.listdir(folder_path):
    match = True#pattern.match(filename)
    if match:
        # # Extract components from the filename
        # prefix, number, sh, cam = match.groups()

        # # Pad number to be 3 digits long
        # new_number = f"{int(number):03d}"

        # # Construct the new filename
        # new_filename = f"{prefix}{new_number}_{sh}_{cam}.avi"
        
        new_filename = filename.replace(' ', '_').lower()

        # Get the full paths
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')
