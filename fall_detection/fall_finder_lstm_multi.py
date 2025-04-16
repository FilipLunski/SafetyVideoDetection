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
from KeypointClassifierLSTMLightningMulti import KeypointClassifierLSTMLightningMulti

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


pose_model = YOLO("../pose_models/yolov8n-pose.pt")


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def main(video_folder, out_folder="", input_format="mp4", seconds_before=2, seconds_after=2, treshold=0.5, lstm_timestamps=30, device="cuda"):
    global labels

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and device != "cpu" else "cpu")

    global fall_model
    fall_model = KeypointClassifierLSTMLightningMulti()
    fall_model.load("model_lstm_multi.pt")

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


def processFile(file, out_folder, seconds_before, seconds_after, treshold, lstm_timestamps, device):

    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("Error reading video file")
        return

    filename = os.path.basename(file)
    dot_index = filename.rfind('.')
    filename_without_extension = filename[:dot_index]
    print(filename_without_extension, end="", flush=True)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    buffer_clear = deque(maxlen=fps * seconds_before)
    buffer_labelled = deque(maxlen=fps * seconds_after)

    buffer_keypoints = deque()

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
    t = 0
    while cap.isOpened():
        success, frame = cap.read()
        start = time.time()
        if success:

            results = pose_model(
                frame, show=False, verbose=False, tracker=None)

            state = -1
            if results[0].keypoints.has_visible == True:
                for keypoints in results[0].keypoints.xy:
                    if device != "cpu":
                        keypoints = keypoints.cpu()
                    normalized_keypoints = normalize_keypoints(
                        keypoints.numpy())
                    buffer_keypoints.append(normalized_keypoints)
                    if len(buffer_keypoints) < lstm_timestamps:
                        continue
                    # print(normalized_keypoints)
                    input_tensor = torch.tensor(
                        buffer_keypoints).unsqueeze(0).to(device)
                    state = fall_model.predict(input_tensor)
                    # print(state)

            labelled_frame = results[0].plot()

            if (state >= 0):
                if state == 0:
                    text = 'normal'
                    color = green
                elif state == 1:
                    text = 'fallen'
                    color = orange
                else:
                    text = 'falling'
                    color = orange

            cv2.putText(labelled_frame, text, position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            buffer_clear.append(frame)
            buffer_labelled.append(labelled_frame)

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
                for buffered_frame in buffer_clear:
                    out_video1.write(buffered_frame)

                for buffered_frame in buffer_labelled:
                    out_video2.write(buffered_frame)
                buffer_clear.clear()
                buffer_labelled.clear()
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
        stop = time.time()
        t += stop-start
    print(f"\tAverage time: {t/frame_number:.2f}s")

    cap.release()
    if out_video1 is not None:
        out_video1.release()
    cv2.destroyAllWindows()
    return (frame_number, t)


# main(r'samples\50ways', r'samples\50ways\50ways_labels.json')

main(r'samples\video\cauca\test', "samples\\out", "avi", 3, 2, 0.3)

# main('samples\\video\\cauca\\test', "samples\\video\\cauca\\out", "avi")
