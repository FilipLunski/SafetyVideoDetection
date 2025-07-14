from ultralytics import YOLO
import cv2
import glob
import json
import os
import numpy as np
import h5py
import torch
import logging
import time
from collections import Counter


logging.basicConfig(
    filename=f'app_{time.time()}.log',
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)


def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data




def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


def main(video_folder, labels_file, out_filename, input_format="mp4", annotated_folder=None, new_file=True):
    global annotations
    annotations = parse_json(labels_file)

    if annotated_folder is not None and not os.path.exists(annotated_folder):
        os.makedirs(annotated_folder)

    if new_file and os.path.exists(out_filename):
        os.remove(out_filename)

    for file in glob.glob(video_folder + "\\*." + input_format):
        print('\n', file, end='\t')
        processFile(file, out_filename, annotated_folder)


def processFile(file, out_filename, annotated_folder):
    with h5py.File(out_filename, 'a') as f:

        pose_model = YOLO("./models_pose/yolo11x-pose")
        fileName_ext = os.path.basename(file)
        dot_index = fileName_ext.rfind('.')
        fileName = fileName_ext[:dot_index]


        if (fileName in annotations):
            annotation = annotations[fileName]
        else:
            print(
                f"Warning: No label found for {fileName}. Using default label.")
            annotation = {
                "id": [0],
                "labels": [
                    {
                        "time": 0,
                        "state": 0
                    }
                ]
            }
        label = annotation["labels"]
        bad = annotation.get("bad", 0)
        if bad == 2:
            print(f"Warning: Video {fileName} is marked as bad. Skipping.")
            return
        ids = annotation.get("id", [])
        
        
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            print("Error reading video file")
            return

        video_group = f.create_group(fileName)

        frame_number = -1
        all_id_counter = Counter()
        supposed_id_counter = Counter({id_: 0 for id_ in ids})

        if annotated_folder is not None:
            annotated_file = os.path.join(annotated_folder, fileName_ext)
            video = cv2.VideoWriter(annotated_file, cv2.VideoWriter_fourcc(
                *'XVID'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        dataset_keypoints = []
        dataset_categories = []

        


        state_number = -1
        state = 0

        if (state_number == len(label) - 1):
            next_state_time = float('inf')
        else:
            next_state_time = label[state_number + 1]["time"]

        frames_found = 0

        while cap.isOpened():
            frame_number += 1
            success, frame = cap.read()
            if success:
                # Get the current position of the video file in milliseconds
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_s = timestamp_ms / 1000.0

                results = pose_model.track(
                    frame, persist=True, tracker="bytetrack.yaml", show=False, verbose=False)
                # print(f"\t {timestamp_s:.2f}", end=' ', flush=True)

                if annotated_folder is not None:
                    frame = results[0].plot()
                if results[0].keypoints.has_visible == False:
                    pass

                # print(results[0].boxes.id, end='', flush=True)
                elif results[0].boxes.id is None or len(results[0].boxes.id) == 0:
                    pass

                else:
                    current_ids = results[0].boxes.id.cpu().numpy()
                    all_id_counter.update(current_ids)
                    supposed_id_counter.update([id for id in current_ids if id in ids])

                    id = -1
                    
                    if ids != []:
                        for i in range(len(results[0].boxes.id)):
                            if results[0].boxes.id[i] in ids:
                                id = i
                                break
                    if id != -1 or ids == []:

                        frames_found += 1

                        normalized_keypoints = normalize_keypoints(
                            results[0].keypoints.xy[id].cpu().numpy())
                    
                        if annotated_folder is not None:
                            cv2.putText(frame, f"{"Normal" if state == 0 else "Fall"}",(results[0].boxes.xyxy[id][:2].int() - torch.tensor([0, 20])).tolist(),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2, cv2.LINE_AA)
                            video.write(frame)

                        if timestamp_s >= next_state_time:
                            state_number += 1
                            state = label[state_number]["state"]
                            print(f"\t {timestamp_s:.2f}", end=': ', flush=True)
                            print(f"State: {state}")

                            if (state_number == len(label) - 1):
                                next_state_time = float('inf')
                            else:
                                next_state_time = label[state_number + 1]["time"]
                        dataset_keypoints.append(normalized_keypoints)
                        dataset_categories.append(state == 1)
                    
                if annotated_folder is not None:
                    video.write(frame)

            else:
                break
        if annotated_folder is not None:
            video.release()

        logger.info(f"Processed {fileName_ext} - Frames found: {frames_found}, Total frames: {frame_number + 1}, Bad: {bad}")

        if ids != []:
            logger.info(f"ID counts: {dict(all_id_counter)}, supposed IDs: {dict(supposed_id_counter)}")

            if any(v==0 for v in supposed_id_counter.values()) or max(supposed_id_counter.values()) - max(all_id_counter.values()):
                logger.warning(f"{fileName_ext} incorrect IDs!!!!!!!!!")

        dataset_group = video_group.create_group('dataset')
        dataset_group.create_dataset(
            'keypoints', data=dataset_keypoints, dtype='float32')
        dataset_group.create_dataset(
            'categories', data=dataset_categories, dtype='bool')

        metadata_group = video_group.create_group('metadata')
        metadata_group.create_dataset(
            'filename', data=np.bytes_(fileName_ext))
        metadata_group.create_dataset('total_frames', data=frame_number)
        metadata_group.create_dataset('bad', data=bad)

        cap.release()
        cv2.destroyAllWindows()

# main('samples\\video\\cauca\\train',
#      'samples\\labels\\caucafall_labels.json', "samples\\dataset_cauca_x_train.h5", "avi")
# main('samples\\video\\cauca\\test',
#      'samples\\labels\\caucafall_labels.json', "samples\\dataset_cauca_x_test.h5", "avi")
# main('samples\\video\\cauca\\validation',
#      'samples\\labels\\caucafall_labels.json', "samples\\dataset_cauca_x_val.h5", "avi")


# main('samples\\video\\fifty_ways\\train',
#      'samples\\labels\\50ways_labels.json', "samples\\dataset_fifty_ways_x_train.h5", "mp4")
# main('samples\\video\\fifty_ways\\test',
#      'samples\\labels\\50ways_labels.json', "samples\\dataset_fifty_ways_x_test.h5", "mp4")
# main('samples\\video\\fifty_ways\\validation',
#      'samples\\labels\\50ways_labels.json', "samples\\dataset_fifty_ways_x_val.h5", "mp4")


# main('samples\\video\\MCFD\\train',
#      'samples\\labels\\mcfd_labels.json', "samples\\dataset_mcfd_x_train.h5", "mp4", "samples\\video\\MCFD\\ann")

# main('samples\\video\\MCFD\\test',
#      'samples\\labels\\mcfd_labels.json', "samples\\dataset_mcfd_x_test.h5", "mp4", "samples\\video\\MCFD\\ann")

# main('samples\\video\\MCFD\\val',
#      'samples\\labels\\mcfd_labels.json', "samples\\dataset_mcfd_x_val.h5", "mp4", "samples\\video\\MCFD\\ann")


main('samples\\video\\Le2i\\train',
     'samples\\labels\\le2i_labels.json', "samples\\dataset_le2i_x_train.h5", "mp4", "samples\\video\\Le2i\\ann")

main('samples\\video\\Le2i\\test',
     'samples\\labels\\le2i_labels.json', "samples\\dataset_le2i_x_test.h5", "mp4", "samples\\video\\Le2i\\ann")

main('samples\\video\\Le2i\\val',
     'samples\\labels\\le2i_labels.json', "samples\\dataset_le2i_x_val.h5", "mp4", "samples\\video\\Le2i\\ann")