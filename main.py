from ultralytics import YOLO
import cv2
import glob
import json
import os
import keyboard
import torch
import numpy as np
from KeypointClassifier import KeypointClassifier

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


pose_model = YOLO("yolov8n-pose.pt")


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def main(video_folder=None, out_folder="", input_format="mp4", video_files = []):
    global fall_model
    fall_model = KeypointClassifier()
    fall_model.load("model_basic.pt")

    if (out_folder == ""):
        out_folder = video_folder + "\\out"
    print(out_folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if len(video_files) > 0:
        files = video_files
    elif video_folder is not None:
        files = glob.glob(video_folder + "\\*." + input_format)
        print (files)
        if len(files) == 0:
            print(f"No .{input_format} files found in {video_folder}")
            return
    else:
        print("No files provided")
        return

    for file in files:
        # print(file)
        processFile(file, out_folder)


def processFile(file, out_folder):

    cap = cv2.VideoCapture(file)

    fileName = os.path.basename(file)
    if not cap.isOpened():
        print("Error reading video file")
        return

    dot_index = fileName.rfind('.')
    outFile = os.path.join(out_folder, fileName[:dot_index] + "_out.avi")
    print(outFile)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    # You can use other codecs like 'MJPG', 'DIVX', etc.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outFile, fourcc, frame_rate,
                          (frame_width, frame_height))
    # print("Frame width:", frame_width)
    # print("Frame height:", frame_height)
    # print("Frame rate:", frame_rate)

    frame_number = 0

    text = 'normal'
    position = (50, 50)  # (x, y) coordinates
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Get the current position of the video file in milliseconds
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = timestamp_ms / 1000.0  # Convert to seconds

            results = pose_model(
                frame, show=False, verbose=False, tracker=None, )

            if results[0].keypoints.has_visible == False:
                continue
            normalized_keypoints = normalize_keypoints(
                results[0].keypoints.xy[0].numpy())
            state = fall_model(torch.from_numpy(
                np.array([normalized_keypoints])))

            if state < 0.5:
                text = 'normal'
                color = green

            elif state < 1.5:
                text = 'fallen'
                color = orange
            # elif state == 1:
            #     text = 'falling'
            #     color = orange
            # elif state == 2:
            #     text = 'fallen'
            #     color = red
            else:
                text = 'normal'
                color = green

            cv2.putText(frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            out.write(frame)
        else:
            break

        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# main(r'samples\50ways', r'samples\50ways\50ways_labels.json')

main("samples\\video\\fifty_ways\\out", video_files=['samples\\source\\people.mp4'])

# main('samples\\video\\fifty_ways\\test',
#      "samples\\video\\fifty_ways\\out", "mp4")

# main('samples\\video\\cauca\\test', "samples\\video\\cauca\\out", "avi")
