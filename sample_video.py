import cv2
import os

path_to_vid = r'data_in\WIN_20261027_11_27_34_Pro.mp4'
out_path = 'calib_data'
os.makedirs(out_path, exist_ok=True)

cap = cv2.VideoCapture(path_to_vid)

fps = cap.get(cv2.CAP_PROP_FPS)

interval = max(1, round(fps * 1.5))
frame_number = 0
saved = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % interval == 0:
            filename = os.path.join(out_path, f"{saved}.png")
            if not cv2.imwrite(filename, frame):
                raise RuntimeError(f"Could not write {filename}")
            saved += 1

        frame_number += 1
finally:
    cap.release()

print(f"Saved {saved} images")