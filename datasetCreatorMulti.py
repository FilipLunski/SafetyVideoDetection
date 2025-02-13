from ultralytics import YOLO
import cv2
import glob
import json
import os
import numpy as np
import h5py
from KeypointClassifier import KeypointClassifier

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


pose_model = YOLO("yolov8s-pose.pt")


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def main(video_folder, labels_file, out_filename, input_format="mp4", new_file=True):
    global labels
    labels = parse_json(labels_file)

    if new_file and os.path.exists(out_filename):
        os.remove(out_filename)

    for file in glob.glob(video_folder + "\\*." + input_format):
        print('\n', file, end = '\t')
        processFile(file, out_filename)


def processFile(file, out_filename):
    with h5py.File(out_filename, 'a') as f:
        # results = model.predict(fileName, show=False, stream=False, save=True)
        # return results
        cap = cv2.VideoCapture(file)

        fileName_ext = os.path.basename(file)
        if not cap.isOpened():
            print("Error reading video file")
            return
        dot_index = fileName_ext.rfind('.')
        fileName = fileName_ext[:dot_index]

        video_group = f.create_group(fileName)

        frame_number = 0

        dataset_keypoints = []
        dataset_categories = []

        if (fileName in labels):
            label = labels[fileName]
        else:
            label = [
                {
                    "time": 0,
                    "state": 0
                }
            ]

        state_number = -1
        state = 0

        if (state_number == len(label) - 1):
            next_state_time = float('inf')
        else:
            next_state_time = label[state_number + 1]["time"]

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Get the current position of the video file in milliseconds
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_s = timestamp_ms / 1000.0  # Convert to seconds

                results = pose_model(frame, show=False, verbose=False)
                print('.', end='', flush=True)

                if results[0].keypoints.has_visible == False:
                    continue

                normalized_keypoints = normalize_keypoints(
                    results[0].keypoints.xy[0].cpu().numpy())

                # print(normalized_keypoints)
                if timestamp_s >= next_state_time:
                    state_number += 1
                    state = label[state_number]["state"]
                    if (state_number == len(label) - 1):
                        next_state_time = float('inf')
                    else:
                        next_state_time = label[state_number + 1]["time"]
                dataset_keypoints.append(normalized_keypoints)
                dataset_categories.append(state)

                # print(normalized_keypoints)

                # print("Keypoint Data (x, y, confidence):", results[0].keypoints.data)

            else:
                break

            frame_number += 1
        dataset_group = video_group.create_group('dataset')
        dataset_group.create_dataset('keypoints', data=dataset_keypoints, dtype='float32')
        dataset_group.create_dataset('categories', data=dataset_categories, dtype='int8')

        metadata_group = video_group.create_group('metadata')
        metadata_group.create_dataset(
            'filename', data=np.bytes_(fileName_ext))
        metadata_group.create_dataset('total_frames', data=frame_number)

        cap.release()
        cv2.destroyAllWindows()


# main(r'samples\50ways', r'samples\50ways\50ways_labels.json')

 
main('samples\\video\\cauca\\train',
     'samples\\labels\\caucafall_labels.json', "samples\\dataset_cauca_multi_train.h5", "avi")
main('samples\\video\\cauca\\test',
     'samples\\labels\\caucafall_labels.json', "samples\\dataset_cauca_multi_test.h5", "avi")
main('samples\\video\\cauca\\validation',
     'samples\\labels\\caucafall_labels.json', "samples\\dataset_cauca_multi_validation.h5", "avi")


main('samples\\video\\fifty_ways\\train',
     'samples\\labels\\50ways_labels.json', "samples\\dataset_fifty_ways_multi_train.h5", "mp4")
main('samples\\video\\fifty_ways\\test',
     'samples\\labels\\50ways_labels.json', "samples\\dataset_fifty_ways_multi_test.h5", "mp4")
main('samples\\video\\fifty_ways\\validation',
     'samples\\labels\\50ways_labels.json', "samples\\dataset_fifty_ways_multi_validation.h5", "mp4")
