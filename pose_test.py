import torch
import torchvision
import cv2
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import pandas as pd
import matplotlib
import glob
from ultralytics import YOLO
import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import logging
import tensorflow as tf


logging.basicConfig(
    filename=f'app_{time.time()}.log',
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, model_type, version):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model_type = model_type
        self.models = {
            "torchvision": (self.torchvision_initialize, self.torchvision_process_image),
            "yolo": (self.yolo_initialize, self.yolo_process_image),
            "mediapipe": (self.mediapipe_initialize, self.media_pipe_process_image)
        }
        self.model_type = model_type
        self.version = version
        self.model_initialize, self.process_image = self.models[model_type]
        self.model_initialize()

    edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)
    ]

    def draw_keypoints(self, poses, scores, frame):
        # the `outputs` is list which in-turn contains the dictionaries
        for i in range(len(poses)):
            keypoints = poses[i]
            # proceed to draw the lines if the confidence score is above 0.9
            if scores[i] > 0.9:
                # keypoints = keypoints[:, :].reshape(-1, 3)
                for p in range(keypoints.shape[0]):
                    # draw the keypoints
                    cv2.circle(frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                               3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    # uncomment the following lines if you want to put keypoint number
                    # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                for ie, e in enumerate(self.edges):
                    # get different colors for the edges
                    rgb = matplotlib.colors.hsv_to_rgb([
                        ie/float(len(self.edges)), 1.0, 1.0
                    ])
                    rgb = rgb*255
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(frame, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                             (int(keypoints[e, 0][1]),
                              int(keypoints[e, 1][1])),
                             tuple(rgb), 2, lineType=cv2.LINE_AA)
            else:
                continue
        return frame

    def torchvision_initialize(self):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                            num_keypoints=17)
        self.model.to(self.device).eval()

    def torchvision_process_image(self, frame, frame_width, frame_height, timestamp):

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Transform the frame
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        frame_tensor = transform(frame_pil)
        # Make the prediction
        with torch.no_grad():
            prediction = self.model([frame_tensor.to(self.device)])

        # Get the keypoints
        keypoints = [pose.cpu().detach().numpy()
                     for pose in prediction[0]['keypoints']]
        scores = prediction[0]['scores']

        return keypoints, scores

    def yolo_initialize(self):
        versions = {
            "nano": "yolo11n-pose.pt",
            "small": "yolo11s-pose.pt",
            "medium": "yolo11m-pose.pt",
            "large": "yolo11l-pose.pt",
            "xlarge": "yolo11x-pose.pt"
        }

        if self.version == "":
            self.version = "nano"

        model_path = versions[self.version]
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def yolo_process_image(self, frame, frame_width, frame_height, timestamp):
        results = self.model(frame, show=False, verbose=False, tracker=None)
        keypoints = [pose.keypoints.xy[0].cpu().numpy() for pose in results]
        scores = [int(pose.keypoints.has_visible) for pose in results]
        return keypoints, scores

    def mediapipe_initialize(self):
        versions = {
            "lite": "pose_landmarker_lite.task"
            "full": "pose_landmarker_full.task",
            "heavy": "pose_landmarker_heavy.task",
        }

        if self.version == "":
            self.version = "lite"

        model_path = versions[self.version]

        num_poses = 4
        min_pose_detection_confidence = 0.5
        min_pose_presence_confidence = 0.5
        min_tracking_confidence = 0.5

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,

            # result_callback=print_result
        )

        landmarker = vision.PoseLandmarker.create_from_options(options)

        self.model = landmarker.detect

        self.keypoints_filter = [0, 1, 4, 7, 8, 11,
                                 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30]

    def media_pipe_process_image(self, frame, frame_width, frame_height, timestamp):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb)

        results = self.model(mp_image)

        if (results.pose_landmarks is None):
            return [], []
        a = [landmarks for landmarks in results.pose_landmarks]

        keypoints = np.array([[[point.x * frame_width, point.y * frame_height] for point in
                               [landmark[i] for i in range(len(landmark)) if i in self.keypoints_filter]] for landmark in results.pose_landmarks])
        scores = [1 for _ in range(len(keypoints))]
        return keypoints, scores

    def process_video(self, filename, out_folder):
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print("Error reading video file")
            logger.error("Error reading video file")
            return

        filename_without_extension = os.path.basename(
            filename)[:filename.rfind('.')]

        logger.info(filename_without_extension)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_number = 0
        t = 0

        out_folder = os.path.join(out_folder, self.model_type)

        out_file = os.path.join(
            out_folder, f"{filename_without_extension}_{self.model_type}_{self.version}.avi")

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(out_file, fourcc, frame_rate,
                                    (frame_width, frame_height))
        # print(out_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_number += 1

            start = time.time()
            keypoints, scores = self.process_image(
                frame, frame_width, frame_height, timestamp_ms)

            stop = time.time()
            t += stop-start

            output_frame = self.draw_keypoints(keypoints, scores, frame)

            if frame_number % 10 == 0:
                print(".", end="", flush=True)
                # print(f"Frame {frame_number}:\tTime: {t/frame_number:.2f}s")

            out_video.write(output_frame)

        out_video.release()
        cap.release()
        logger.info(f"Frames: {frame_number}:\tTime: {t/frame_number:.2f}s")
        return frame_number, t


def main(video_folders, out_folder, model_type, version="", input_formats=["mp4", "avi"]):
    
    print(f"\nTest for {model_type} {version}........................................................................................")
    logger.info(f"Test for {model_type} {version}........................................................................................")

    if (out_folder == ""):
        out_folder = "\\out"
    logger.info(out_folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    files = []
    for folder in video_folders:
        for input_format in input_formats:
            files += glob.glob(folder + "\\*." + input_format)

    if len(files) == 0:
        print("No files found")
        logger.error("No files found")

    pose_estimator = PoseEstimator(model_type, version)
    
    frame_number = 0
    tim = 0
    
    for file in files:
        f,t = pose_estimator.process_video(file, out_folder)
        frame_number += f
        tim += t
    
    logger.info(f"Average time/frame for {model_type} {version}:\t{tim/frame_number:.4f}s, {frame_number/tim}fps")

# main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "torchvision", version="")

main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "mediapipe", version="lite")
main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "mediapipe", version="full")
main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "mediapipe", version="heavy")

# main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "yolo", version="nano")
# main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "yolo", version="small")
# main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "yolo", version="medium")
# main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "yolo", version="large")
# main([r'samples\video\fifty_ways\test', r'samples\video\cauca\test'], "samples\\out", "yolo", version="xlarge")

