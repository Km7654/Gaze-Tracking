# 👁️ Eye Tracking & Blink-Based Mouse Control

This Python application enables **hands-free control of your computer** using **eye movement for cursor control** and **blinks for clicking**, using only a **webcam**. It leverages **MediaPipe** for face and eye landmark detection, and trains a **regression model** to map your eye position to screen coordinates.

---

## 🚀 Features

- 🎯 Real-time **eye tracking** for cursor movement  
- 👁️‍🗨️ Blink detection for **click actions**:
  - **Triple blink** → Left Click
  - **Quadruple blink** → Right Click
- 🧠 Machine learning-based cursor prediction using **Ridge Regression** with polynomial features
- 📐 One-time **calibration** to adapt the model to your face and camera setup
- 📸 Only requires a **webcam**

---

## 🛠️ Requirements

Install required packages using pip:

```bash
pip install opencv-python mediapipe pyautogui scikit-learn joblib numpy
```

---

## 📦 Files

- `eye_tracking.py` — Main script
- `eye_tracker_model.joblib` — Saved model after calibration (auto-generated)
- `README.md` — Project documentation

---

## 🧪 First-Time Setup (Calibration Required)

Before tracking can begin, you must **calibrate** the system so it learns how your eye positions map to screen coordinates.

### ✅ Steps:

1. Run the script:
   ```bash
   python eye_tracking.py
   ```
2. If no model is found, calibration will start automatically.
3. **Follow the instructions on screen** — move your eyes to the cursor position as it moves.
4. Wait 7 seconds per target location while the system collects data.
5. Once complete, the model is saved automatically.

> ⚠️ **Important**: Do not move your head during calibration. Only move your eyes!

---

## 🕹️ Usage After Calibration

After the model is trained:

- Just run:
  ```bash
  python eye_tracking.py
  ```
- The cursor will follow your eye movement.
- Blink 3 times quickly to **left click**, or 4 times to **right click**.

---

## 📏 Optional: Evaluate Model Accuracy

To get feedback on how accurate the model is, you can modify the script to print:

```python
from sklearn.metrics import mean_absolute_error, r2_score

pred = model.predict(X_calib)
print("MAE:", mean_absolute_error(y_calib, pred))
print("R2 Score:", r2_score(y_calib, pred))
```

---

## 💡 Tips

- Use in **well-lit** environments for better face and eye detection.
- Ensure the webcam is at **eye level** and your **face remains stable**.
- Re-run calibration if you change your environment or lighting.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
