import cv2
import time

ip_addresses = [
    # '10.202.161.217',
    '10.202.161.219',
    '10.202.161.221',
    '10.202.161.222',
    '10.202.161.223',
    '10.202.161.225',
    '10.202.161.226'
]


def connect_to_camera(url, max_retries=1, retry_delay=0.5):
    for _ in range(max_retries):
        cap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        time.sleep(retry_delay)
    return None


i = 0

while True:
    ip_address = ip_addresses[i]
    i += 1
    print(ip_address)
    url = f'http://user:mobotix@{ip_address}/control/faststream.jpg?needlength&stream=MxPEG&preview&previewsize=640x480&quality=40&fps=4'
    cap = connect_to_camera(url)

    if not cap.isOpened():
        print("Error reading video file")
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video1 = cv2.VideoWriter(f"out_file_{ip_address}.avi", fourcc, frame_rate,
                                     (frame_width, frame_height))
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_count += 1
                out_video1.write(frame)
            else:
                print("Stream interrupted. Attempting to reconnect...")
                cap.release()
                cap = connect_to_camera(url)
                print(frame_count)

                continue

        print(frame_count)
        cap.release()
        out_video1.release()
