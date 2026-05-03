# Hebrew Sign Language Recognition 🤟

Real-time recognition of Israeli Sign Language (ISL) letters using a webcam.

## Demo
*Coming soon*

## How It Works

1. **MediaPipe** detects 21 landmarks on the hand in each frame
2. The 21 landmarks (x, y, z each) are normalized relative to the wrist → **63 features**
3. A **Random Forest** classifier predicts which Hebrew letter is being signed
4. Predictions are smoothed over 7 frames to reduce noise

## Results

- **22 Hebrew letters** recognized
- **99.77% accuracy** on test set
- Dataset: **4,400 images** collected and labeled manually

## Project Structure

```
├── collect_data.py     # Capture training images from webcam
├── train_model.py      # Extract landmarks and train the classifier
├── recognize.py        # Real-time recognition via webcam
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt
```

**Collect data** (200 images per letter):
```bash
python collect_data.py
```

**Train the model:**
```bash
python train_model.py
```

**Run real-time recognition:**
```bash
python recognize.py
```

## Tech Stack

- Python
- OpenCV
- MediaPipe (Google)
- scikit-learn
- NumPy / Pillow
