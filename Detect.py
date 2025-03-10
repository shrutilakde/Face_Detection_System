import os
import cv2
import numpy as np
import face_recognition
import pickle

# Define dataset path
dataset_path = "employees"

known_encodings = []
known_names = []

# Loop through employee folders
for employee_name in os.listdir(dataset_path):
    employee_folder = os.path.join(dataset_path, employee_name)
    
    if os.path.isdir(employee_folder):  # Ensure it's a folder
        print(f"Processing: {employee_name}")
        
        # Loop through images in the folder
        for img_name in os.listdir(employee_folder):
            img_path = os.path.join(employee_folder, img_name)
            image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face and extract encoding
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:  # Ensure face was found
                known_encodings.append(encodings[0])
                known_names.append(employee_name)

# Save encodings to a pickle file
data = {"encodings": known_encodings, "names": known_names}
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Face encodings saved in 'face_encodings.pkl' successfully!")