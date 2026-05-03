import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import urllib.request

LETTERS = [
    ('א', 'alef'), ('ב', 'bet'), ('ג', 'gimel'), ('ד', 'dalet'),
    ('ה', 'he'), ('ו', 'vav'), ('ז', 'zayin'), ('ח', 'het'),
    ('ט', 'tet'), ('י', 'yod'), ('כ', 'kaf'), ('ל', 'lamed'),
    ('מ', 'mem'), ('נ', 'nun'), ('ס', 'samech'), ('ע', 'ayin'),
    ('פ', 'pe'), ('צ', 'tsadi'), ('ק', 'kof'), ('ר', 'resh'),
    ('ש', 'shin'), ('ת', 'tav')
]

TARGET = 200
DATA_DIR = 'data'
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print('Downloading hand landmarker model (~30MB)...')
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print('Done.')


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 100, 0), -1)


def load_font(size):
    for path in ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/David.ttf', 'C:/Windows/Fonts/calibri.ttf']:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_hebrew(frame, text, x, y, font, color=(255, 210, 0)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])


def main():
    download_model()
    os.makedirs(DATA_DIR, exist_ok=True)
    font_big = load_font(80)

    cap = cv2.VideoCapture(0)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    idx = 0
    prev_idx = -1
    count = 0
    folder = ''
    heb, name = LETTERS[0]
    capturing = False
    last_save = 0

    while 0 <= idx < len(LETTERS):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if idx != prev_idx:
            heb, name = LETTERS[idx]
            folder = os.path.join(DATA_DIR, name)
            count = count_images(folder)
            capturing = False
            prev_idx = idx

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        hand_ok = len(result.hand_landmarks) > 0

        # Save BEFORE drawing skeleton so training images are clean
        now = time.time()
        if capturing and hand_ok and (now - last_save) > 0.08:
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, f'{count}.jpg'), frame)
            last_save = now
            count += 1

        if hand_ok:
            draw_hand(frame, result.hand_landmarks[0])

        if count >= TARGET:
            idx += 1
            continue

        # --- UI ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        frame = draw_hebrew(frame, heb, w // 2 - 35, 5, font_big)

        cv2.putText(frame, f'{idx + 1}/{len(LETTERS)}  ({name})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        bx1, bx2 = 10, w - 10
        by1, by2 = h - 48, h - 30
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), -1)
        fill = int((bx2 - bx1) * count / TARGET)
        cv2.rectangle(frame, (bx1, by1), (bx1 + fill, by2), (0, 200, 80), -1)
        cv2.putText(frame, f'{count} / {TARGET}', (bx1, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if capturing and hand_ok:
            border = (0, 255, 0)
            msg = 'CAPTURING  |  SPACE = pause'
        elif capturing:
            border = (0, 140, 255)
            msg = 'Waiting for hand...'
        elif hand_ok:
            border = (0, 220, 220)
            msg = 'Hand ready  |  SPACE = start  |  N = next  |  B = back  |  Q = quit'
        else:
            border = (0, 0, 200)
            msg = 'No hand  |  SPACE = start  |  N = next  |  B = back  |  Q = quit'

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border, 4)
        cv2.putText(frame, msg, (10, h - 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, border, 2)

        cv2.imshow('ISL - Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:
            capturing = not capturing
        elif key == ord('n'):
            idx += 1
        elif key == ord('b'):
            idx = max(0, idx - 1)

    cap.release()
    cv2.destroyAllWindows()

    print('\n=== Collection Summary ===')
    for h_char, h_name in LETTERS:
        c = count_images(os.path.join(DATA_DIR, h_name))
        status = 'done' if c >= TARGET else f'{c}/{TARGET}'
        print(f'  {h_char}  {h_name:<10} {status}')


if __name__ == '__main__':
    main()
