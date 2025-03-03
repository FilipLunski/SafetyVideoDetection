import torch
import torchvision
import cv2
import argparse
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
import re
import pandas as pd
import pickle
import matplotlib
import glob
from ultralytics import YOLO
import os

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]


def draw_keypoints(poses, scores, frame):
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
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(frame, (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                         (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                         tuple(rgb), 2, lineType=cv2.LINE_AA)
        else:
            continue
    return frame


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torchvision_initialize():
    global model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                   num_keypoints=17)
    model.to(device).eval()

def torchvision_process_image(frame):

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
        prediction = model([frame_tensor.to(device)])

    # Get the keypoints
    keypoints = [pose.cpu().detach().numpy() for pose in prediction[0]['keypoints']]
    scores = prediction[0]['scores']
    
    return keypoints, scores



def yolo_initialize():
    global model
    model = YOLO("yolov8n-pose.pt")

def yolo_process_image(frame):
    results = model(frame, show=False, verbose=False, tracker=None)
    keypoints = [pose.keypoints.xy[0].cpu().numpy() for pose in results]
    scores = [int(pose.keypoints.has_visible) for pose in results]
    return keypoints, scores




def process_video(filename, out_folder, process_image):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Error reading video file")
        return

    filename_without_extension = os.path.basename(
        filename)[:filename.rfind('.')]

    print(filename_without_extension)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_number = 0


    out_file = os.path.join(
        out_folder, f"{filename_without_extension}_out.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_file, fourcc, frame_rate,
                                (frame_width, frame_height))
    print(out_file)
    t = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        start = time.time()
        keypoints, scores = process_image(frame)

        stop = time.time()
        t += stop-start

        output_frame = draw_keypoints(keypoints, scores, frame)

        if frame_number % 10 == 0:
            print(f"Frame {frame_number}:\tTime: {t/frame_number:.2f}s")

        out_video.write(output_frame)

    out_video.release()
    cap.release()

    return 


models = {
    "torchvision": (torchvision_initialize, torchvision_process_image),
    "yolo": (yolo_initialize, yolo_process_image)
}


def main(video_folder, out_folder, model_type, input_format="mp4"):
    model_initialize, process_image = models[model_type]

    if (out_folder == ""):
        out_folder = video_folder + "\\out"
    print(out_folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    files = glob.glob(video_folder + "\\*." + input_format)

    if len(files) == 0:
        print("No files found")

    model_initialize()

    for file in files:
        process_video(file, out_folder, process_image)


main(r'samples\video\fifty_ways\test', "samples\\out", "yolo", "mp4")
