import os
import numpy as np
from PIL import Image
import cv2
import pickle

"""This file face_recognition is responsible for training the face cascade model which will later be used in the 
liveVideo.py file for implementing face detection and recognition with labels. The file also creates labels with 
corresponding faces for use in the file discussed above."""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "photos")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml') # cascade model being used
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    # loop through files found in directory
    for file in files:
        # check for correct file type
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # takes name of directory containing image
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            # add new label if it doesn't exist already
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")  # convert to grayscale
            size = (250, 250)# resize image for performance
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")  # create numpy array from the converted grayscale image
            # check for faces in numpy array
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=6)
            # gets the region of the face on the image and appends it to the two arrays
            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

with open('pickles/labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
