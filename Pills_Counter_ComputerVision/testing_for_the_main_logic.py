import cv2
import numpy as np
def count_pills_optimized(image_path):
    """
    Counts whole and broken pills in an image using the Hough Circle Transform.

    Args:
        image_path: The path to the image file.

    Returns:
        A tuple containing:
        - The processed image with detected pills marked.
        - The count of whole pills.
        - The count of broken pills.
    """
    # Load the image
    frame = cv2.imread('image.png')
    if frame is None:
        return None, 0, 0

    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a median blur to reduce noise, which is important for Hough Transform
    blur = cv2.medianBlur(gray, 5)

    # Use the Hough Circle Transform to detect circular shapes
    # The parameters are tuned for the provided image.
    # You might need to adjust them for different images.
    circles = cv2.HoughCircles(
        blur,                       # Source image
        cv2.HOUGH_GRADIENT,         # Detection method
        dp=1.2,                     # Inverse ratio of the accumulator resolution to the image resolution
        minDist=50,                 # Minimum distance between the centers of detected circles
        param1=50,                  # Higher threshold for the Canny edge detector
        param2=30,                  # Accumulator threshold for circle centers
        minRadius=20,               # Minimum circle radius
        maxRadius=40                # Maximum circle radius
    )

    whole_pills = 0
    broken_pills = 0
    
    # Check if any circles were found
    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))
        whole_pills = len(circles[0])
        
        # Draw the detected circles on the original image
        for (x, y, r) in circles[0, :]:
            # Draw the outer circle in green
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # Draw the center of the circle in red
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    # Calculate broken pills. We assume any detected object that isn't a whole pill is a broken piece.
    # This requires a second pass using contour detection to count the remaining objects.
    # This part of the code is simplified and might need further refinement for specific use cases.
    # For this specific image, the main goal is to count the whole pills correctly.
    
    # We'll use the original contour-based method to find remaining objects
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_contours = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:  # Filter out small noise
            total_contours += 1

    # Broken pills are the total contours minus the whole pills we found with HoughCircles
    # This is an approximation as some contours might be noise or part of the same pill
    broken_pills = total_contours - whole_pills
    if broken_pills < 0:
        broken_pills = 0 # Ensure the count doesn't go negative

    return frame, whole_pills, broken_pills

# Main execution block
image_path = "image.jpg" # Make sure the image is in the same directory
processed_image, whole_count, broken_count = count_pills_optimized(image_path)

if processed_image is not None:
    # Display the counts on the processed image
    cv2.putText(processed_image, f"Whole Pills: {whole_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(processed_image, f"Broken Pills: {broken_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the result
    cv2.imshow("Optimized Pill Counter", processed_image)
    cv2.waitKey(0) # Wait indefinitely for a key press
    cv2.destroyAllWindows()
else:
    print("Error: Could not load the image.")