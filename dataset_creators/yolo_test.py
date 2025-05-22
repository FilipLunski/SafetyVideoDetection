from ultralytics import YOLO
import os
import glob


pose_model = YOLO("./models_pose/yolo11x-pose")


def main(video_folder, out_video_folder, input_format="mp4"):

    if (not os.path.exists(out_video_folder)):
        os.makedirs(out_video_folder)

    for file in glob.glob(video_folder + "\\*." + input_format):
        print('\n', file, end='\t')
        pose_model.track(file, tracker="bytetrack.yaml",
                         show=False, verbose=False, save=True)



main('samples\\video\\MCFD',
     'samples\\video\\MCFD\\output', "mp4")