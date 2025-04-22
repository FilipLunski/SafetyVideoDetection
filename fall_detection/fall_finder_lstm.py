from ultralytics import YOLO
import cv2
import glob
import json
import os
# import keyboard
import torch
import numpy as np
from collections import deque
import time
from KeypointClassifierLSTMLightning import KeypointClassifierLSTMLightning
from FallFinder import FallFinder

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


pose_model = "../models_pose/yolov8s-pose.pt"


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def processFile(file, out_folder, seconds_before, seconds_after, treshold, lstm_timestamps, device):

    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error reading video file")
        return

    filename = os.path.basename(file)
    dot_index = filename.rfind('.')
    filename_without_extension = filename[:dot_index]
    print(filename_without_extension,"------------------------------")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_filename = os.path.join(
        out_folder, f"{filename_without_extension}_out_{"{}"}.avi")
    output_annotated_filename = os.path.join(
        out_folder, f"a_{filename_without_extension}_out_{"{}"}.avi")
    print(output_filename, output_annotated_filename)

    fall_finder = FallFinder(pose_model, fall_model, lstm_timestamps, frame_width,
                             frame_height, frame_rate, output_filename, output_annotated_filename)

    frame_number = 0
    t = 0

    while cap.isOpened():
        success, frame = cap.read()
        start = time.time()
        if success:
            fall_finder.process_frame(frame)
        else:
            break

        frame_number += 1
        stop = time.time()
        t += stop-start
    fall_finder.release_video_writers()
    print(f"\tAverage time: {t/frame_number:.2f}s")

    cap.release()

    cv2.destroyAllWindows()
    return (frame_number, t)


def main(video_folder, out_folder="", input_format="mp4", seconds_before=2, seconds_after=2, treshold=0.5, lstm_timestamps=50, device="cuda"):
    global labels

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and device != "cpu" else "cpu")

    global fall_model  
    fall_model = KeypointClassifierLSTMLightning.load_from_checkpoint(
        "./logs/lstm_50_1_64_64_0.4_0.4/version_0/checkpoints/epoch=399-step=2000.ckpt")

    if (out_folder == ""):
        out_folder = video_folder + "\\out"
    print(out_folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    files = glob.glob(video_folder + "\\*." + input_format)

    if len(files) == 0:
        print("No files found")

    time_all = 0
    frames_all = 0

    for file in files:
        # print(file)
        (f, t) = processFile(file, out_folder, seconds_before=2, seconds_after=2,
                             treshold=treshold, lstm_timestamps=lstm_timestamps, device=device)
        time_all += t
        frames_all += f

    print(f"Average time: {time_all/frames_all:.2f}s")


# main(r'samples\50ways', r'samples\50ways\50ways_labels.json')
main(r'samples\video\cauca\test', "samples\\out\\lstm", "avi", 3, 2, 0.5)
main(r'samples\video\fifty_ways\test', "samples\\out\\lstm", "mp4", 3, 2, 0.5)

# main('samples\\video\\cauca\\test', "samples\\video\\cauca\\out", "avi")
