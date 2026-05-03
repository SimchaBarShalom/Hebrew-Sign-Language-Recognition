import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import time
from PIL import Image, ImageDraw, ImageFont

MODEL_FILE = 'sign_model.pkl'
MODEL_PATH = 'hand_landmarker.task'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


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


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 100, 0), -1)


def get_features(landmarks):
    wrist = landmarks[0]
    coords = []
    for lm in landmarks:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return coords


def main():
    print('Loading model...')
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    letters = data['letters']
    print(f'Model loaded. Recognizes {len(letters)} letters.\n')

    font_letter = load_font(120)
    font_small = load_font(26)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    # Smoothing: keep last N predictions and pick most common
    history = []
    HISTORY_SIZE = 7

    cv2.namedWindow('ISL - Sign Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ISL - Sign Recognition', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        hand_ok = len(result.hand_landmarks) > 0

        predicted_heb = ''
        predicted_name = ''
        confidence = 0.0

        if hand_ok:
            draw_hand(frame, result.hand_landmarks[0])
            features = get_features(result.hand_landmarks[0])
            proba = model.predict_proba([features])[0]
            label_idx = int(np.argmax(proba))
            confidence = proba[label_idx]

            history.append(label_idx)
            if len(history) > HISTORY_SIZE:
                history.pop(0)

            # Use the most common prediction in history (smoothing)
            stable_idx = max(set(history), key=history.count)
            predicted_heb, predicted_name = letters[stable_idx]

        # --- UI ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 45), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        if hand_ok and predicted_heb:
            frame = draw_hebrew(frame, predicted_heb, 20, 0, font_letter, (255, 210, 0))

            bar_w = int((w - 200) * confidence)
            cv2.rectangle(frame, (160, 20), (w - 20, 50), (50, 50, 50), -1)
            bar_color = (0, 220, 80) if confidence > 0.7 else (0, 140, 255)
            cv2.rectangle(frame, (160, 20), (160 + bar_w, 50), bar_color, -1)
            cv2.putText(frame, f'{predicted_name}  {confidence:.0%}',
                        (160, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                          (0, 220, 80) if confidence > 0.7 else (0, 140, 255), 4)

        cv2.imshow('ISL - Sign Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
