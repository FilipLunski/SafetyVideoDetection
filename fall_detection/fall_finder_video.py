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

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


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
                                 frame_height, frame_rate, output_filename, output_annotated_filename, threshold=threshold, print_statistics=False)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            fall_detector.process_frame(frame)
        else:
            break

    fall_detector.release_video_writers()
    print(f"\tTotal average time: {fall_detector.avg_inf_time:.4f}s")

    cap.release()

    cv2.destroyAllWindows()
    return (fall_detector._frames_count, fall_detector._total_time)


def main(video_folder, out_folder="", input_format="mp4", seconds_before_after=2, threshold=0.5, rnn_timestamps=50, device="cuda", fall_device="cpu"):
    global labels

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and device != "cpu" else "cpu")

    global fall_model
    if (rnn_timestamps == 1):
        fall_model = KeypointClassifierFFNN.load_from_checkpoint(
            "logs_m_fnn2\\ffnn_[34, 128, 64, 32]_0.3_relu\\version_0\\checkpoints\\epoch=499-step=2500.ckpt")
    else:
        fall_model = KeypointClassifierGRU.load_from_checkpoint(
            "logs_m\\gru_50_1_64_64_0.15_0.4\\version_91\\checkpoints\\epoch=549-step=2750.ckpt")
    fall_model.to(device)
    if (fall_device == "cpu"):
        fall_model.cpu()
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

    print(f"Average time: {time_all/frames_all:.4f}s")


# main(r'samples\video\fifty_ways\test', "samples\\out\\gruu", "mp4", 3, 0.5, 50)
# main(r'samples\video\cauca\test', "samples\\out\\gruu", "avi", 3, 0.5, 50)

# main(r'samples\video\fifty_ways\test', "samples\\out\\basicc", "mp4", 3, 0.5, 1)
# main(r'samples\video\cauca\test', "samples\\out\\basicc", "avi", 3, 0.5, 1)



main(r'samples/video/MPFDD', "samples\\out\\mpfdd\\gru", "mp4", 3, 0.5, 50, "cuda")
# main(r'samples/video/MPFDD', "samples\\out\\mpfdd\\ffnn", "mp4", 3, 0.5, 1, "cuda")

