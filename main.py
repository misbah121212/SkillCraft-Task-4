import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------
# Step 1: Load model and labels
# -----------------------------
model = load_model("model/hand_gesture_model.h5")
labels = np.load("model/labels.npy", allow_pickle=True)

# -----------------------------
# Step 2: Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

print("✅ Webcam started! Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Flip horizontally for natural feel
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI) for hand
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (128, 128))
    roi_norm = roi_resized / 255.0
    roi_expanded = np.expand_dims(roi_norm, axis=0)

    # -----------------------------
    # Step 3: Make prediction
    # -----------------------------
    predictions = model.predict(roi_expanded)
    class_index = np.argmax(predictions)
    gesture_name = labels[class_index]
    confidence = predictions[0][class_index]

    # -----------------------------
    # Step 4: Display result
    # -----------------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{gesture_name} ({confidence*100:.2f}%)"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Program closed.")
