import cv2
from ultralytics import YOLO

# Load the trained model (correctly handling the file name with spaces and parentheses)
model = YOLO('/Users/ishansharma/Downloads/best (1).pt')

# Check if model loaded successfully
if model is None:
    print("Error loading the model.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the webcam opens successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection
    results = model(frame, verbose=False)  # Disable verbose output for cleaner display
    print(results)  # Log results for debugging

    # Annotate the frame with detection results
    annotated_frame = results[0].plot() if results else frame  # Fallback to original frame if no results

    # Resize frame for better display
    frame_resized = cv2.resize(annotated_frame, (640, 480))

    # Display the annotated frame
    cv2.imshow('Sign Language Detection', frame_resized)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
