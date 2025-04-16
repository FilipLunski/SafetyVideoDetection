from ultralytics import YOLO
import cv2
import glob
import json
import os
import keyboard
import torch
import numpy as np
import threading
from collections import deque
from KeypointClassifier import KeypointClassifier


ip_addresses = [
    # '10.202.161.217',
    '10.202.161.219',
    # '10.202.161.221',
    '10.202.161.222',
    '10.202.161.223',
    '10.202.161.225',
    '10.202.161.226'
]

caps = {}



green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


pose_model = YOLO("../pose_models/yolov8m-pose.pt")


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def processCamera(i):

    ip_address = ip_addresses[i]
    ip_address_without = ip_address.replace(".", "_")
    out_folder = 'samples\\video\\out\\cam_'+ip_address_without
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print(out_folder)
    seconds_before = 5
    seconds_after = 10
    treshold = 0.4

    caps[ip_address] = cv2.VideoCapture(
        f'http://user:mobotix@{ip_address}/control/faststream.jpg?needlength&stream=MxPEG&previewsize=640x480&quality=40&fps=4')

    if not caps[ip_address].isOpened():
        print("Error reading video file")
        return

    frame_width = int(caps[ip_address].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[ip_address].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = 4  # int(cap.get(cv2.CAP_PROP_FPS))
    print(frame_rate)

    buffer1 = deque(maxlen=frame_rate * seconds_before)
    buffer2 = deque(maxlen=frame_rate * seconds_after)
    frames_after = frame_rate * seconds_after

    # Define the codec and create a VideoWriter object
    # You can use other codecs like 'MJPG', 'DIVX', etc.
    # print("Frame width:", frame_width)
    # print("Frame height:", frame_height)
    # print("Frame rate:", frame_rate)

    frame_number = 0
    frames_left = 100
    out_video1 = None
    out_video2 = None
    out_video_number = 0

    text = 'normal'
    position = (50, 50)  # (x, y) coordinates
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2

    check = 0

    while caps[ip_address].isOpened():
        success, frame = caps[ip_address].read()

        if success:

            results = pose_model(
                frame, show=False, verbose=False, tracker=None)

            state = 0
            labelled_frame = frame.copy()
            if results[0].keypoints.has_visible == True:
                for keypoints in results[0].keypoints.xy:
                    normalized_keypoints = normalize_keypoints(
                        keypoints.numpy())
                    # print(normalized_keypoints)
                    state = fall_model(torch.from_numpy(
                        np.array([normalized_keypoints])))

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

            frames_left = frames_after if state == 1  else (  #or results[0].keypoints.has_visible == True
                0 if frames_left <= 0 else frames_left - 1)

            if state == 1 or frames_left > 0:
                if out_video1 is None:
                    out_file1 = os.path.join(
                        out_folder, f"cam_{ip_address_without}_out_{out_video_number}.avi")
                    print(f"{ip_address_without}: Creating new video: {out_file1}")

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out_video1 = cv2.VideoWriter(out_file1, fourcc, frame_rate,
                                                 (frame_width, frame_height))

                    out_file2 = os.path.join(
                        out_folder, f"cam_{ip_address_without}_out_{out_video_number}_l.avi")
                    out_video2 = cv2.VideoWriter(out_file2, fourcc, frame_rate,
                                                 (frame_width, frame_height))

                    out_video_number += 1
                for buffered_frame in buffer1:
                    out_video1.write(buffered_frame)

                for buffered_frame in buffer2:
                    out_video2.write(buffered_frame)
                buffer1.clear()
                buffer2.clear()
                # print('.')
            elif out_video1 is not None:
                print(f"{ip_address_without}: Releasing video")
                out_video1.release()
                out_video1 = None
                out_video2.release()
                out_video2 = None
                # print('...')

                # Print the frame number and timestamp
                # print(f"Frame {frame_number}: {timestamp_s:.2f} seconds")
            check += 1
            if (check >= 100):
                check = 0
                print(f"{ip_address_without} running")
        else:
            break

        frame_number += 1

    caps[ip_address].release()
    if out_video1 is not None:
        out_video1.release()
    cv2.destroyAllWindows()


def main():
    # processCamera(ip_addresses[0])

    # List to store threads
    threads = []

    # Create and start threads with different parameters
    for i in range(len(ip_addresses)):
        thread = threading.Thread(target=processCamera, args=[i])
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

fall_model = KeypointClassifier()
fall_model.load("model_basic.pt")


main()

# processCamera(0)