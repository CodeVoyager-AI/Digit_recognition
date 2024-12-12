import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("D:\Digit _classification_project\_classification_project\mnist_model.h5")  # Replace with the path to your model

cap = cv2.VideoCapture(0)

print("Press 's' to capture and predict, 'q' to quit.")

predicted_digit = None

# Define the box dimensions
box_start = (200, 100)  # Top-left corner (x, y)
box_end = (400, 300)    # Bottom-right corner (x, y)
box_color = (255, 0, 0)  # Blue box
box_thickness = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Draw the bounding box on the live feed
    display_frame = frame.copy()
    cv2.rectangle(display_frame, box_start, box_end, box_color, box_thickness)

    # If a prediction exists, overlay it on the frame
    if predicted_digit is not None:
        cv2.putText(display_frame, f"Predicted Digit: {predicted_digit}",
                    (50, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam - Predict Digits", display_frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  # Press 's' to capture and predict
        # Crop the region inside the box
        roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]

        # Preprocess the cropped region
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        inverted = cv2.bitwise_not(resized)  # Ensure colors match MNIST
        _, thresholded = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
        normalized = thresholded / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)

        # Predict digit
        predictions = model.predict(reshaped)
        print("Prediction probabilities:", predictions)
        predicted_digit = np.argmax(predictions)
        print(f"Predicted Digit: {predicted_digit}")

    elif key == ord('q'):  # Press 'q' to quit
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
