import cv2
import os
import time

# Step 1: Get Employee Name and Create Folder
employee_name = input("Enter Employee Name: ")
save_path = f"employees/{employee_name}"
os.makedirs(save_path, exist_ok=True)

# Step 2: Initialize Camera and Face Detector
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
side_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")  # For better detection

count = 0
max_images = 50

print("\n📸 Starting Face Data Collection... Look into the camera.")

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not detected. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using multiple classifiers
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:  # Try profile face detection if frontal fails
        faces = side_face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Save in grayscale
        face_resized = cv2.resize(face, (150, 150))  # Resize for consistency

        img_path = f"{save_path}/img{count}.jpg"
        cv2.imwrite(img_path, face_resized)
        count += 1

        # Draw rectangle and display progress
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Images Captured: {count}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Delay to prevent capturing too fast
        time.sleep(0.2)

    cv2.imshow("Face Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n⏹ Capture Stopped by User.")
        break

cap.release()
cv2.destroyAllWindows()

if count == max_images:
    print("\n✅ Face Data Collection Complete: All images saved successfully!")
else:
    print(f"\n⚠ Face Data Collection Incomplete: {count}/{max_images} images saved.")