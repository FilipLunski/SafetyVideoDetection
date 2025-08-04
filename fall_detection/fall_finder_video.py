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
from pathlib import Path


pose_model = "./models_pose/yolo11x-pose.pt"
folders = [
    "samples/video/le2i/test",
    "samples/video/fifty_ways/test",
    "samples/video/cauca/test",
    "samples/video/mcfd/test",
    "samples/video/MPFDD",
]


def processFile(file, out_folder, seconds_before_after, threshold, lstm_timestamps, device):

    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error reading video file")
        return

    filename = os.path.basename(file)
    dot_index = filename.rfind('.')
    filename_without_extension = filename[:dot_index]
    print(file, "------------------------------")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    step = round(fps / 10)
    if step < 1:
        step = 1
    print(f"Step: {step}")

    output_filename = os.path.join(
        out_folder, f"{filename_without_extension}_out_{"{}"}.avi")
    output_annotated_filename = os.path.join(
        out_folder, f"a_{filename_without_extension}_out_{"{}"}.avi")

    fall_detector = FallDetector(pose_model, fall_model, lstm_timestamps, frame_width,
                                 frame_height, round(fps/step), output_filename, output_annotated_filename, threshold=threshold, print_statistics=False, frames_buffer_size=500)

    i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            i = i % step + 1
            if i == step:
                fall_detector.process_frame(frame)
        else:
            break

    fall_detector.release_video_writers()
    print(f"\tAverage time: {fall_detector.avg_inf_time:.4f}s")

    cap.release()

    cv2.destroyAllWindows()
    return (fall_detector._frames_count, fall_detector._total_time)


def main(out_folder="", input_format=["mp4", "avi"], fall_model_path="", seconds_before_after=2, threshold=0.5, rnn_timestamps=50, device="cuda"):
    global labels

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and device != "cpu" else "cpu")

    global fall_model
    if (rnn_timestamps == 1):
        fall_model = KeypointClassifierFFNN.load_from_checkpoint(
            "fall_detection/FFNN.ckpt")
    else:
        fall_model = KeypointClassifierGRU.load_from_checkpoint(
            fall_model_path)
    fall_model.to(device)
    fall_model.eval()

    model_name = next((part for part in Path(
        fall_model_path).parts if part.startswith("gru_")), None)
    out_folder = os.path.join(out_folder, model_name)

    print(out_folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    all_files = []

    for ext in input_format:
        for folder in folders:
            pattern = os.path.join(folder, f"*.{ext}")
            files = glob.glob(pattern)
            all_files.extend(files)

    if len(all_files) == 0:
        print("No files found")

    time_all = 0
    frames_all = 0

    for file in all_files:
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

# main("samples\\out\\GRU\\neeew", ["mp4","avi"], r"tb_logs\logs_gru_64_64\gru_65_1_64_64_0.15_0.5_6144\version_0\checkpoints\epoch=0437.ckpt", 5, 0.5, 65)

# main("samples\\out\\GRU\\neeew", ["mp4","avi"], r"tb_logs\logs_gru\gru_50_1_128_64_0.15_0.4\version_5\checkpoints\epoch=499-step=10500.ckpt", 3, 0.5, 50)
# main("samples\\out\\GRU\\neeew", ["mp4","avi"], r"tb_logs\logs_gru\gru_60_1_64_64_0.15_0.4_8192\version_0\checkpoints\epoch=549-step=9900.ckpt", 3, 0.5, 60)
# main("samples\\out\\GRU\\neeew", ["mp4","avi"], r"tb_logs\logs_gru\gru_65_1_64_64_0.15_0.4_6144\version_53\checkpoints\epoch=719-step=16560.ckpt", 3, 0.5, 65)
# main("samples\\out\\GRU\\neeew", ["mp4","avi"], r"tb_logs\logs_gru\gru_70_1_64_64_0.15_0.4_4096\version_0\checkpoints\epoch=549-step=19250.ckpt", 3, 0.5, 70)
main("samples\\out\\GRU\\neeew", ["mp4", "avi"], r"tb_logs\logs_gru_64_64\gru_22_1_64_64_0.15_0.4_6144\version_0\checkpoints\epoch=0448.ckpt", 3, 0.5, 22)
