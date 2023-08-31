import cv2
import os

# Load the vehicle detection classifier (Haarcascades)
vehicle_cascade = cv2.CascadeClassifier('D:/My AIT Knowledge/Deep Learning Computer Vision/Lec1/HW_LEC1/haarcascade_car.xml')

# Initialize the camera
cap = cv2.VideoCapture(" rtsp:ad")
#cap = cv2.VideoCapture("rtsp://admin")

# Create a directory to save images
save_directory = "vehicle_images"
os.makedirs(save_directory, exist_ok=True)

# Capture 100 images
image_count = 0
#image_count_cam1 = 0

while image_count < 200:
    ret,  frame  = cap.read()
 #   ret1, frame1 = cap1.read()
    
    if not ret:
        break
 #   if not ret1:
 #       break

    # Convert the frame to grayscale for detection
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #   gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Detect vehicles
 #   vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))

    # Draw rectangles around detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
     # Draw rectangles around detected vehicles1
 #   for (x, y, w, h) in vehicles1:
 #       cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the image with rectangles if vehicles are detected
    if len(vehicles) > 0:
        image_count += 1
        image_filename = os.path.join(save_directory, f"vehicle_{image_count}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"Image '{image_filename}' saved with {len(vehicles)} vehicle(s) detected.")
    
    # Save the image with rectangles if vehicles1 are detected
 #   if len(vehicles1) > 0:
 #       image_count_cam1 += 1
 #       image_filename = os.path.join(save_directory, f"vehicle_{image_count_cam1}.jpg")
 #       cv2.imwrite(image_filename, frame1)
 #       print(f"Image '{image_filename}' saved with {len(vehicles1)} vehicle(s) detected.")

    cv2.imshow("Vehicle Detection", frame)
 #   cv2.imshow("Vehicle Detection1", frame1)

    # Press Esc key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
#cap1.release()
cv2.destroyAllWindows()
