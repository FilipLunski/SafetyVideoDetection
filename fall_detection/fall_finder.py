from ultralytics import YOLO
import cv2
import glob
import json
import os
import torch
import numpy as np
from collections import deque
from KeypointClassifier import KeypointClassifier

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


pose_model = YOLO("../pose_models/yolov8n-pose.pt")


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def main(video_folder, out_folder="", input_format="mp4", seconds_before=2, seconds_after=2, treshold=0.5, device="cuda"):

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and device != "cpu" else "cpu")

    global labels

    global fall_model
    fall_model = KeypointClassifier()
    fall_model.load("model_basic.pt")

    if (out_folder == ""):
        out_folder = video_folder + "\\out"
    print(out_folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    files = glob.glob(video_folder + "\\*." + input_format)

    if len(files) == 0:
        print("No files found")

    for file in files:
        # print(file)
        processFile(file, out_folder, seconds_before=2,
                    seconds_after=2, treshold=treshold, device=device)


def processFile(file, out_folder, seconds_before, seconds_after, treshold, device):

    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error reading video file")
        return

    filename = os.path.basename(file)
    dot_index = filename.rfind('.')
    filename_without_extension = filename[:dot_index]
    print(filename_without_extension)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    buffer1 = deque(maxlen=fps * seconds_before)
    buffer2 = deque(maxlen=fps * seconds_after)
    frames_after = fps * seconds_after

    # Define the codec and create a VideoWriter object
    # You can use other codecs like 'MJPG', 'DIVX', etc.
    # print("Frame width:", frame_width)
    # print("Frame height:", frame_height)
    # print("Frame rate:", frame_rate)

    frame_number = 0
    frames_left = 0
    out_video1 = None
    out_video2 = None
    out_video_number = 0

    text = 'normal'
    position = (50, 50)  # (x, y) coordinates
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2

    while cap.isOpened():
        success, frame = cap.read()

        if success:

            results = pose_model(
                frame, show=False, verbose=False, tracker=None)

            state = 0
            if results[0].keypoints.has_visible == True:
                for keypoints in results[0].keypoints.xy:
                    if device != "cpu":
                        keypoints = keypoints.cpu()
                    normalized_keypoints = normalize_keypoints(
                        keypoints.numpy())
                    # print(normalized_keypoints)
                    normalized_keypoints
                    inn = torch.tensor([normalized_keypoints]).to(device)
                    state = fall_model(inn)

                    if state >= treshold:
                        state = 1
            labelled_frame = results[0].plot()
            if state < 0.5:
                text = 'normal'
                color = green
            elif state < 1.5:
                text = 'fallen'
                color = orange
            else:
                text = 'normal'
                color = green
            cv2.putText(labelled_frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            buffer1.append(frame)
            buffer2.append(labelled_frame)

            frames_left = frames_after if state == 1 else (
                0 if frames_left <= 0 else frames_left - 1)

            if state == 1 or frames_left > 0:
                if out_video1 is None:
                    out_file1 = os.path.join(
                        out_folder, f"{filename_without_extension}_out_{out_video_number}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out_video1 = cv2.VideoWriter(out_file1, fourcc, frame_rate,
                                                 (frame_width, frame_height))

                    out_file2 = os.path.join(
                        out_folder, f"{filename_without_extension}_out_{out_video_number}_l.avi")
                    out_video2 = cv2.VideoWriter(out_file2, fourcc, frame_rate,
                                                 (frame_width, frame_height))

                    out_video_number += 1
                for buffered_frame in buffer1:
                    out_video1.write(buffered_frame)

                for buffered_frame in buffer2:
                    out_video2.write(buffered_frame)
                buffer1.clear()
                buffer2.clear()
            elif out_video1 is not None:
                out_video1.release()
                out_video1 = None
                out_video2.release()
                out_video2 = None

                # Print the frame number and timestamp
                # print(f"Frame {frame_number}: {timestamp_s:.2f} seconds")

        else:
            break

        frame_number += 1

    cap.release()
    if out_video1 is not None:
        out_video1.release()
    cv2.destroyAllWindows()


# main(r'samples\50ways', r'samples\50ways\50ways_labels.json')

main(r'samples\video\cauca\test', "samples\\out\\basic", "avi", 3, 2)
main(r'samples\video\fifty_ways\test', "samples\\out\\basic", "mp4", 3, 2)

# main('samples\\video\\cauca\\test', "samples\\video\\cauca\\out", "avi")
