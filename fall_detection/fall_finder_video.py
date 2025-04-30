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
                                 frame_height, frame_rate, None, None, threshold=threshold, print_statistics=False)

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


main(r'samples\video\fifty_ways\test', "samples\\out\\gruu", "mp4", 3, 0.5, 50)
main(r'samples\video\cauca\test', "samples\\out\\gruu", "avi", 3, 0.5, 50)

main(r'samples\video\fifty_ways\test', "samples\\out\\basicc", "mp4", 3, 0.5, 1)
main(r'samples\video\cauca\test', "samples\\out\\basicc", "avi", 3, 0.5, 1)



# main(r'samples\video\fifty_ways\test', "samples\\out\\gruu", "mp4", 3, 0.5, 50, "cuda")
# main(r'samples\video\cauca\test', "samples\\out\\gruu", "avi", 3, 0.5, 50, "cuda")

# main(r'samples\video\fifty_ways\test', "samples\\out\\basicc", "mp4", 3, 0.5, 1, "cuda")
# main(r'samples\video\cauca\test', "samples\\out\\basicc", "avi", 3, 0.5, 1, "cuda")



""" 
cuda
samples\out\gruu
50ways-seg01 ------------------------------
        Average time: 0.0546s
50ways-seg09 ------------------------------
        Average time: 0.0663s
50ways-seg17 ------------------------------
        Average time: 0.0549s
50ways-seg21 ------------------------------
        Average time: 0.0522s
50ways-seg25 ------------------------------
        Average time: 0.0599s
50ways-seg33 ------------------------------
        Average time: 0.0509s
50ways-seg41 ------------------------------
        Average time: 0.0511s
50ways-seg49 ------------------------------
        Average time: 0.0517s
Average time: 0.0552s
samples\out\gruu
FallBackwardsS01 ------------------------------
        Average time: 0.0484s
FallForwardS02 ------------------------------
        Average time: 0.0622s
FallForwardS05 ------------------------------
        Average time: 0.0467s
FallLeftS06 ------------------------------
        Average time: 0.0384s
FallRightS03 ------------------------------
        Average time: 0.0498s
FallRightS07 ------------------------------
        Average time: 0.0486s
FallSittingS08 ------------------------------
        Average time: 0.0446s
HopS04 ------------------------------
        Average time: 0.0553s
HopS09 ------------------------------
        Average time: 0.0519s
KneelS10 ------------------------------
        Average time: 0.0470s
PickupobjectS03 ------------------------------
        Average time: 0.0451s
PickupobjectS05 ------------------------------
        Average time: 0.0598s
SitDownS02 ------------------------------
        Average time: 0.0457s
WalkS03 ------------------------------
        Average time: 0.0453s
WalkS06 ------------------------------
        Average time: 0.0372s
Average time: 0.0478s

samples\out\basicc
50ways-seg01 ------------------------------
        Average time: 0.0589s
50ways-seg09 ------------------------------
        Average time: 0.0530s
50ways-seg17 ------------------------------
        Average time: 0.0548s
50ways-seg21 ------------------------------
        Average time: 0.0515s
50ways-seg25 ------------------------------
        Average time: 0.0517s
50ways-seg33 ------------------------------
        Average time: 0.0612s
50ways-seg41 ------------------------------
        Average time: 0.0548s
50ways-seg49 ------------------------------
        Average time: 0.0570s
Average time: 0.0553s
samples\out\basicc
FallBackwardsS01 ------------------------------
        Average time: 0.0591s
FallForwardS02 ------------------------------
        Average time: 0.0680s
FallForwardS05 ------------------------------
        Average time: 0.0589s
FallLeftS06 ------------------------------
        Average time: 0.0441s
FallRightS03 ------------------------------
        Average time: 0.0456s
FallRightS07 ------------------------------
        Average time: 0.0475s
FallSittingS08 ------------------------------
        Average time: 0.0379s
HopS04 ------------------------------
        Average time: 0.0464s
HopS09 ------------------------------
        Average time: 0.0650s
KneelS10 ------------------------------
        Average time: 0.0474s
PickupobjectS03 ------------------------------
        Average time: 0.0445s
PickupobjectS05 ------------------------------
        Average time: 0.0464s
SitDownS02 ------------------------------
        Average time: 0.0458s
WalkS03 ------------------------------
        Average time: 0.0453s
WalkS06 ------------------------------
        Average time: 0.0363s
Average time: 0.0480s

 """

""" 
cpu

 """