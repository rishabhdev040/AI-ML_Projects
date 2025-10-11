import cv2
import numpy as np
import math

def count_pills(frame):
    
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use adaptive thresholding to create a binary image.
    # This is robust to uneven lighting and highlights objects on a dark background.
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Clean up small noise and separate touching objects using morphological operations.
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours, which are boundaries of detected objects.
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    whole_pills = 0
    broken_pills = 0

    # Iterate through each detected contour to analyze its shape
    for c in contours:
        # Filter out very small contours that are likely just noise
        area = cv2.contourArea(c)
        if area > 100:
            # Get the bounding rectangle for aspect ratio calculation
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            
            # Check if the contour is convex (no inward curves)
            is_convex = cv2.isContourConvex(c)

            # Define thresholds for whole pills based on shape.
            # You may need to adjust these values based on the specific pill shapes.
            # A whole pill, regardless of shape, should be relatively convex.
            # A broken piece often has a low aspect ratio or is not convex.
            if is_convex and 0.5 <= aspect_ratio <= 2.0:
                whole_pills += 1
                color = (0, 255, 0)  # Green for whole pills
            else:
                broken_pills += 1
                color = (0, 0, 255)  # Red for broken pills

            # Draw a bounding rectangle around the detected pill
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame, whole_pills, broken_pills

# Initialize video capture from the default webcam
cap = cv2.VideoCapture('image.png')

# Main loop to read frames and process them
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the current frame
    result, whole_pill_count, broken_pill_count = count_pills(frame)
    # Display the counts on the frame
    cv2.putText(result, f"Whole Pills: {whole_pill_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, f"Broken Pills: {broken_pill_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the final result
    cv2.imshow("Pill Counter", result)

    # Press 'q' to exit the loop
    if cv2.waitKey(50000) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()