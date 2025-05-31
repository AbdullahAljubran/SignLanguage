import cv2
import os
import time
from datetime import datetime

def capture_images():
    """
    Captures 10 images from webcam with 5-second intervals
    Saves images to 'new_images' folder
    """
    
    # Create directory if it doesn't exist
    output_dir = "new_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Camera will use its default resolution
    
    print("Camera initialized. Starting capture in 5 seconds...")
    print("Press 'q' during preview to quit early")
    
    # Initial 5-second sleep
    time.sleep(5)
    
    # Show initial preview for a few seconds before starting capture
    print("Showing preview window - make sure you can see yourself...")
    for i in range(30):  # Show preview for 3 seconds
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Image Capture - Press q to quit', frame)
            cv2.waitKey(100)
    
    images_captured = 0
    target_images = 10
    
    try:
        while images_captured < target_images:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Use frame without resizing
            
            # Display the frame so you can see yourself
            cv2.imshow('Image Capture - Press q to quit', frame)
            cv2.waitKey(1)  # Allow window to refresh
            
            # Generate filename (simple numbering from 1 to 10)
            filename = f"{images_captured + 1}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            success = cv2.imwrite(filepath, frame)
            
            if success:
                images_captured += 1
                print(f"Captured image {images_captured}/{target_images}: {filename}")
            else:
                print(f"Error: Failed to save image {images_captured + 1}")
            
            # Wait for 5 seconds before next capture with live preview
            if images_captured < target_images:
                print(f"Waiting 5 seconds before next capture...")
                
                # Show live preview during the 5-second wait
                start_time = time.time()
                while time.time() - start_time < 5.0:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow('Image Capture - Press q to quit', frame)
                    
                    key = cv2.waitKey(30) & 0xFF  # 30ms refresh rate
                    if key == ord('q'):
                        print("Quit requested by user")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
        
        print(f"\nCapture complete! {images_captured} images saved in '{output_dir}' folder")
        print("Images are saved in camera's native resolution")
        
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    print("=== Image Capture Program ===")
    print("This program will:")
    print("- Wait 5 seconds before starting")
    print("- Capture 10 images at camera's native resolution")
    print("- Wait 5 seconds between each capture")
    print("- Save images to 'new_images' folder")
    print("- Show live preview (press 'q' to quit early)")
    print("\nStarting capture...")
    
    capture_images()