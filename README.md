# Face_Detection_System
Training Phase: train.py Purpose: Build a dataset and train a facial recognition model.
Steps:

Capture 50 images per person from multiple angles to ensure the model learns robust facial features for each individual.
Store the captured images in a structured format, creating the training dataset.
Train the recognition model using these images. The trained model learns unique facial characteristics for each employee, preparing it for detection tasks.
Detection Phase: detect.py Purpose: Perform real-time face detection using the trained dataset.
Steps:

Load the trained dataset into the detection script.
Utilize the YOLO library for real-time, efficient face detection. YOLO is a deep learning framework known for its speed and accuracy in object detection tasks.
Process video streams or images to detect faces and match them with the trained dataset to identify individuals.
Monitoring and Output Phase: monitore.py Purpose: Provide the final output of the project, including real-time face monitoring and reporting.
Steps:

Capture live video feed using a webcam.
Use the previously trained model and YOLO to detect faces in real-time.
Match detected faces with the trained dataset to identify employees.
Determine the employees' working status (whether they are active or not).
Generate a detailed report:

List of detected employees.
Their working status (working or not working).
Save the data into an Excel file for documentation and further analysis.
