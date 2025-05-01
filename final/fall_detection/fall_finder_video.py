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
from KeypointClassifierLSTM import KeypointClassifierLSTM
from KeypointClassifierGRU import KeypointClassifierGRU
from KeypointClassifierFFNN import KeypointClassifierFFNN
from FallDetector import FallDetector



pose_model = "./models_pose/yolo11m-pose.pt"


def processFile(file, out_folder, seconds_before_after, threshold, lstm_timestamps, device):

    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error reading video file")
        return

    filename = os.path.basename(file)
    dot_index = filename.rfind('.')
    filename_without_extension = filename[:dot_index]
    print(filename_without_extension, "------------------------------")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_filename = os.path.join(
        out_folder, f"{filename_without_extension}_out_{"{}"}.avi")
    output_annotated_filename = os.path.join(
        out_folder, f"a_{filename_without_extension}_out_{"{}"}.avi")

    fall_detector = FallDetector(pose_model, fall_model, lstm_timestamps, frame_width,
                                 frame_height, frame_rate, output_filename, None, threshold=threshold, print_statistics=False)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            fall_detector.process_frame(frame)
        else:
            break

    fall_detector.release_video_writers()
    print(f"\tAverage time: {fall_detector.avg_inf_time:.4f}s")

    cap.release()

    cv2.destroyAllWindows()
    return (fall_detector._frames_count, fall_detector._total_time)


def main(video_folder, out_folder="", input_format="mp4", seconds_before_after=2, threshold=0.5, rnn_timestamps=50, device="cuda"):
    global labels

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and device != "cpu" else "cpu")

    global fall_model
    if (rnn_timestamps == 1):
        fall_model = KeypointClassifierFFNN.load_from_checkpoint(
            "fall_detection/FFNN.ckpt")
    else:
        fall_model = KeypointClassifierGRU.load_from_checkpoint(
            "fall_detection/GRU.ckpt")
    fall_model.to(device)
    fall_model.eval()

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
        (f, t) = processFile(file, out_folder, seconds_before_after=seconds_before_after,
                             threshold=threshold, lstm_timestamps=rnn_timestamps, device=device)
        time_all += t
        frames_all += f

    print(f"Total average time: {time_all/frames_all:.4f}s")



# main(r'samples\video\fifty_ways\test', "samples\\out\\FFNN", "mp4", 3, 0.5, 1)
# main(r'samples\video\cauca\test', "samples\\out\\FFNN", "avi", 3, 0.5, 1)

# main(r'samples\video\fifty_ways\test', "samples\\out\\GRU", "mp4", 3, 0.5, 50)
# main(r'samples\video\cauca\test', "samples\\out\\GRU", "avi", 3, 0.5, 50)


