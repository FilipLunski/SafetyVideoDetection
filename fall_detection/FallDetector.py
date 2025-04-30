import torch
from collections import deque
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
import numpy as np
from torch import tensor
from KeypointClassifierGRU import KeypointClassifierGRU
from KeypointClassifierLSTM import KeypointClassifierLSTM
import cv2
import time

green = (0, 255, 0)
red = (0, 0, 255)
orange = (0, 165, 255)

annotations: dict = {
    0: {
        "color": green,
        "text": "Normal"
    },
    1: {
        "color": red,
        "text": "Fallen"
    }
}


class Person:
    def __init__(self, id, sequence_length):
        self._id = id
        self._sequence_length = sequence_length
        self._buffer_keypoints = deque(maxlen=sequence_length)
        self._state = 0
        self._confidence = 0
        self._bounding_box = [0, 0, 0, 0]
        self._text_position = (50, 50)
        self._unseen_frames = 0
        self._fallen_frames = 0

    @property
    def id(self):
        return self._id  # getter

    @property
    def buffer_keypoints(self):
        return self._buffer_keypoints

    @buffer_keypoints.setter
    def buffer_keypoints(self, value):
        self._buffer_keypoints.append(value)

    @buffer_keypoints.deleter
    def buffer_keypoints(self):
        self._buffer_keypoints.clear()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = value

    @property
    def bounding_box(self):
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value):
        self._bounding_box = value

    @property
    def unseen_frames(self):
        return self._unseen_frames

    @unseen_frames.setter
    def unseen_frames(self, value):
        self._unseen_frames = value

    @property
    def text_position(self):
        return self._text_position

    @text_position.setter
    def text_position(self, value):
        self._text_position = value

    @property
    def fallen_frames(self):
        return self._fallen_frames

    @fallen_frames.setter
    def fallen_frames(self, value):
        self._fallen_frames = value


def normalize_keypoints(keypoints):

    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)

    keypoints[(keypoints[:, 0] == 0) & (keypoints[:, 1] == 0)] = [-1, -1]

    return np.where(keypoints != -1, (keypoints - [x_min, y_min]) / [x_max - x_min, y_max - y_min], keypoints).flatten()


class FallDetector:
    def __init__(self, pose_model, fall_model, sequence_length,
                 frame_width, frame_height, frame_rate,
                 video_output_file=None, annotated_video_output_file=None,
                 frames_buffer_size=200, threshold=0.5, calc_statistics=True, print_statistics=True):
        self._pose_model: Model = YOLO(pose_model)
        self._fall_model: KeypointClassifierGRU | KeypointClassifierLSTM = fall_model
        self._sequence_length = sequence_length
        self._video_output_file: str = video_output_file
        self._annotated_video_output_file: str = annotated_video_output_file
        self._video_number = 0
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._frame_rate = frame_rate
        self._threshold = threshold
        self._frames_buffer_size = frames_buffer_size
        self._frame_buffer = deque(maxlen=frames_buffer_size)
        self._video = None
        self._annotated_frame_buffer = deque(maxlen=frames_buffer_size)
        self._annotated_video = None
        self._frames_left = 0
        self._fall_finder_persons = {}
        self._statistics = calc_statistics
        self._print_statistics = print_statistics

        if(self._statistics):
            self._frames_count = 0
            self._total_time = 0

        # print(self._fall_model)

    @property
    def avg_inf_time(self):
        if self._statistics:
            return self._total_time / self._frames_count
        else:
            return None

    def process_frame(self, frame, release=False):
        if(self._statistics):
            start = time.perf_counter()
        self._frames_left = max(0, self._frames_left - 1)
        for person in self._fall_finder_persons.values():
            person.unseen_frames += 1
        if (self._video != None):
            self._frame_buffer.append(frame)
        results = self._pose_model.track(
            frame, show=False, verbose=False, persist=True)
        result: Results = results[0]

        if result.keypoints.has_visible == False:
            return None
        if (self._annotated_video_output_file != None):
            annotated_frame = result.plot()

        if self._fall_model._device != "cpu":
            result = result.cpu()

        seen_persons: list[Person] = []

        # print(result.keypoints.xy.shape if result.keypoints.xy is not None else None, result.boxes.xywhn.shape if result.boxes.xywhn is not None else None, result.boxes.xyxy.shape if result.boxes.xyxy is not None else None, result.boxes.id.shape if result.boxes.id is not None else None)
        if (result.boxes.id is None):
            print("No IDs detected!!!")
            return
        for keypoints, bounding_box, bounding_box_abs, id in zip(result.keypoints.xy, result.boxes.xywhn, result.boxes.xyxy, result.boxes.id):
            person: Person = None

            keypoints = keypoints.cpu()
            bounding_box = bounding_box.cpu()
            bounding_box_abs = bounding_box_abs.cpu()
            id = id.cpu()

            id = int(id.item())
            if (np.sum(np.all(keypoints.numpy() == 0, axis=1)) < 8):
                normalized_keypoints = normalize_keypoints(keypoints.numpy())
                if (id in self._fall_finder_persons):
                    person = self._fall_finder_persons[id]
                else:
                    person = Person(id, self._sequence_length)
                    self._fall_finder_persons[id] = person
                person.bounding_box = bounding_box
                person.buffer_keypoints = normalized_keypoints
                person.unseen_frames = 0
                person.text_position = self.normalize_text_position(
                    (bounding_box_abs[0], bounding_box_abs[1]))
                self.update_state(person)
                seen_persons.append(person)

                if (self._annotated_video_output_file != None):
                    cv2.putText(annotated_frame, f"{annotations[person.state]['text']} {person.confidence:.2f}", person.text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotations[person.state]["color"], 2)
        if (self._annotated_video_output_file != None):
            self._annotated_frame_buffer.append(annotated_frame)

        if (self._frames_left > 0):
            self.initialize_video_writers()
            if self._video != None:
                while len(self._frame_buffer) > 0:
                    self._video.write(self._frame_buffer.popleft())
            if self._annotated_video != None:
                while len(self._annotated_frame_buffer) > 0:
                    self._annotated_video.write(
                        self._annotated_frame_buffer.popleft())
        if (self._frames_left <= 0 or release):
            self.release_video_writers()
        if(self._statistics):
            end = time.perf_counter()
            if self._print_statistics:
                print(f"{end - start:.6f}\t", end="")
                print(len(self._fall_finder_persons), end=" ")
                for person in self._fall_finder_persons.values():
                    print(f"\t{person.id}: {person.state} {person.unseen_frames} {person.fallen_frames}", end=" ")
                print()
            self._frames_count += 1
            self._total_time += end - start

        return [p.state for p in seen_persons]

    def initialize_video_writers(self):
        if self._video == None and self._video_output_file != None:
            path = self._video_output_file.format(self._video_number)
            self._video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(
                *'XVID'), self._frame_rate, (self._frame_width, self._frame_height))
        if self._annotated_video == None and self._annotated_video_output_file != None:
            path = self._annotated_video_output_file.format(self._video_number)
            self._annotated_video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(
                *'XVID'), self._frame_rate, (self._frame_width, self._frame_height))

        self._video_number += 1

    def release_video_writers(self):
        if self._video != None:
            self._video.release()
            self._video = None
        if self._annotated_video != None:
            self._annotated_video.release()
            self._annotated_video = None

    def update_state(self, person: Person):
        if len(person.buffer_keypoints) >= 1:

            if(self._sequence_length> 1):
                keypoints = np.array(person.buffer_keypoints)
            else:
                keypoints = np.array(person.buffer_keypoints[0])

            input_tensor = tensor(keypoints).unsqueeze(0)
            if self._fall_model._device.type == "cpu":
                input_tensor = input_tensor.cpu()
            else:
                input_tensor = input_tensor.cuda()
            if(self._statistics):
                start = time.perf_counter()
            # print(self._fall_model._device)
            state = self._fall_model.predict_step(input_tensor).item()
            if self._fall_model._device.type != "cpu":
                torch.cuda.synchronize()
            if(self._statistics):
                end = time.perf_counter()
                if self._print_statistics:
                    print(f"{end - start:.6f}\t", end="")
                    print(f"{end - start:.6f}\t", end="")

            # print(state)
            if state >= self._threshold:
                confidence = (state - self._threshold) / (1 - self._threshold)
                state = 1
            else:
                confidence = (self._threshold - state) / self._threshold
                state = 0
            person.state = state
            person.confidence = confidence
            if state == 1:
                self._frames_left = self._frames_buffer_size
                person.fallen_frames += 1
            else:
                person.fallen_frames = 0

    def normalize_text_position(self, position):
        position_x = position[0].item()
        position_y = position[1].item()-20
        position_x = max(5, min(position_x, self._frame_width - 80))
        position_y = max(5, min(position_y, self._frame_height - 30))
        return (int(position_x), int(position_y))
