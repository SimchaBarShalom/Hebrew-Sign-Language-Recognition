import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = 'data'
MODEL_FILE = 'sign_model.pkl'
MODEL_PATH = 'hand_landmarker.task'

LETTERS = [
    ('א', 'alef'), ('ב', 'bet'), ('ג', 'gimel'), ('ד', 'dalet'),
    ('ה', 'he'), ('ו', 'vav'), ('ז', 'zayin'), ('ח', 'het'),
    ('ט', 'tet'), ('י', 'yod'), ('כ', 'kaf'), ('ל', 'lamed'),
    ('מ', 'mem'), ('נ', 'nun'), ('ס', 'samech'), ('ע', 'ayin'),
    ('פ', 'pe'), ('צ', 'tsadi'), ('ק', 'kof'), ('ר', 'resh'),
    ('ש', 'shin'), ('ת', 'tav')
]


def extract_landmarks(image_path, landmarker):
    """
    Reads an image, detects the hand, and returns 63 normalized coordinates.
    Normalization: all points are relative to the wrist (landmark 0),
    so position on screen doesn't affect the features.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    wrist = landmarks[0]

    coords = []
    for lm in landmarks:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    return coords


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    X = []
    y = []
    skipped = 0

    print('Extracting landmarks from images...\n')

    for label_idx, (heb, name) in enumerate(LETTERS):
        folder = os.path.join(DATA_DIR, name)
        if not os.path.exists(folder):
            print(f'  Missing folder: {name}')
            continue

        images = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
        found = 0

        for img_file in images:
            features = extract_landmarks(os.path.join(folder, img_file), landmarker)
            if features:
                X.append(features)
                y.append(label_idx)
                found += 1
            else:
                skipped += 1

        print(f'  {heb} ({name}): {found} samples')

    print(f'\nTotal: {len(X)} samples  |  Skipped (no hand detected): {skipped}')

    if len(X) == 0:
        print('No data found. Make sure the data/ folder has images.')
        return

    X = np.array(X)
    y = np.array(y)

    # Remove letters with too few samples to split properly
    min_samples = 5
    valid_mask = np.array([np.sum(y == i) >= min_samples for i in y])
    X, y = X[valid_mask], y[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f'\nTraining Random Forest on {len(X_train)} samples...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test set: {acc:.2%}\n')

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'letters': LETTERS}, f)
    print(f'Model saved to {MODEL_FILE}\n')

    unique_labels = sorted(set(y_test))
    report_names = [LETTERS[i][1] for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=report_names, zero_division=0))


if __name__ == '__main__':
    main()
