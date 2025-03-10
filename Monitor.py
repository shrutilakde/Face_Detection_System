import cv2
import face_recognition
import pickle
import pandas as pd
import time
import dlib
from datetime import datetime
from ultralytics import YOLO
import threading

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Load face encodings
with open("face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)

# Load dlib face detector & landmark predictor
face_detector = dlib.get_frontal_face_detector()


# Open webcam
cap = cv2.VideoCapture(0)

# Create an empty DataFrame
data = pd.DataFrame(columns=["Room", "Timestamp", "People Detected", "Recognized Faces", "Status"])

# Function to save data periodically
def save_data_periodically():
    global data
    while True:
        if not data.empty:
            data.to_excel("People_Activity_Log.xlsx", index=False)
            print("✅ Data saved at", datetime.now().strftime("%H:%M:%S"))
        time.sleep(10)

# Start a thread for periodic saving
data_saving_thread = threading.Thread(target=save_data_periodically, daemon=True)
data_saving_thread.start()

room_id = 1  # Room number, can be dynamically assigned if multiple cameras are used

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face recognition
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_faces = []
    
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(face_data["encodings"], face_encoding, tolerance=0.5)
        name = "Unknown"
        
        if True in matches:
            match_index = matches.index(True)
            name = face_data["names"][match_index]
        
        recognized_faces.append(name)
        
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Detect people and objects using YOLO
    results = model(frame)
    person_boxes = []
    laptop_boxes, mobile_boxes = [], []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_boxes.append((x1, y1, x2, y2))

            elif label == "laptop":
                laptop_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            elif label == "cell phone":
                mobile_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Filter out people behind others
    filtered_person_boxes = []
    for (x1, y1, x2, y2) in sorted(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        is_behind = False
        for (fx1, fy1, fx2, fy2) in filtered_person_boxes:
            overlap_x = max(0, min(x2, fx2) - max(x1, fx1))
            overlap_y = max(0, min(y2, fy2) - max(y1, fy1))
            overlap_area = overlap_x * overlap_y
            
            # Ignore smaller, significantly overlapped persons
            if overlap_area > 0.5 * ((x2 - x1) * (y2 - y1)):
                is_behind = True
                break
        
        if not is_behind:
            filtered_person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Final foreground person count
    person_count = len(filtered_person_boxes)

    # Determine activity status
    status = "Not Working"

    for (px1, py1, px2, py2) in filtered_person_boxes:
        person_center_x = (px1 + px2) // 2
        person_center_y = (py1 + py2) // 2
        person_height = py2 - py1  # Height of the person

        # Estimate if the person is sitting or standing
        frame_height = frame.shape[0]
        sitting = person_height < 0.6 * frame_height  # If height is less than 60% of the frame, assume sitting
        standing = not sitting

        # Check if the person is near a laptop
        near_laptop = any(
            abs(person_center_x - (lx1 + lx2) // 2) < 100 and abs(person_center_y - (ly1 + ly2) // 2) < 100
            for (lx1, ly1, lx2, ly2) in laptop_boxes
        )

        if sitting and near_laptop:
            status = "Working"
        elif standing and not near_laptop:
            status = "Not Working"

    # Log the recognized faces with activity status
    recognized_faces_str = ", ".join(recognized_faces) if recognized_faces else "No Recognized Faces"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[room_id, timestamp, person_count, recognized_faces_str, status]],
                            columns=["Room", "Timestamp", "People Detected", "Recognized Faces", "Status"])
    data = pd.concat([data, new_data], ignore_index=True)

    # Display the results
    cv2.putText(frame, f"Room {room_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition & Activity Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

# Save final results
data.to_excel("People_Activity_Log.xlsx", index=False)
print("✅ Final data saved to 'People_Activity_Log.xlsx'")

cap.release()
cv2.destroyAllWindows()
