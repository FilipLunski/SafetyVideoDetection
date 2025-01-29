
import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    video = cv2.VideoCapture("people.mp4")

    frame_count = 0
    total_time = 0

    while True:
        success, frame = video.read()

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        start_time = time.time()
        results = pose.process(frame)
        end_time = time.time()

        print(vars(results))
        print("-----------------------------------------")
        # print(results.pose_world_landmarks)

        total_time += (end_time - start_time)
        frame_count += 1

        frame.flags.writeable = True
        mp_drawing.draw_landmarks(frame, results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)



        print("----------------------------------------------------------------------------------")

        

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    average_time_per_frame = total_time / frame_count
    print(f"Average time per frame: {average_time_per_frame:.4f} seconds")