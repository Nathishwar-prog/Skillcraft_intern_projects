import cv2
import os

# Create a dataset folder if it doesn't exist
DATASET_PATH = "dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# Input gesture name
gesture_name = input("Enter the gesture name: ")
gesture_path = os.path.join(DATASET_PATH, gesture_name)
os.makedirs(gesture_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting images for '{gesture_name}'... Press 'q' to stop.")

while count < 1000:  # Collect 1000 images per gesture
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow("Collecting Gesture Data", frame)

    # Save the frame
    img_path = os.path.join(gesture_path, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Dataset collection for '{gesture_name}' completed!")
