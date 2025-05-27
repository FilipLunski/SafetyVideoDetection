from ultralytics import YOLO
import os
import glob
from concurrent.futures import ThreadPoolExecutor


pose_model = YOLO("./models_pose/yolo11x-pose")


def main(video_folder, input_format="mp4"):

    video_files = glob.glob(os.path.join(video_folder, f"*.{input_format}"))

    for file in video_files:
        process_video(file)


def process_video(file):

    print('\n', file, "\t------------------------------------------------------------------------------------------------------------------------------------")
    pose_model.track(file, tracker="bytetrack.yaml",
                     show=False, verbose=False, save=True)


main('samples\\video\\Le2i', "mp4")
