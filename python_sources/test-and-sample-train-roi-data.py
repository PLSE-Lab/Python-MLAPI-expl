"""
    ATTENTION: THIS SCRIPT WILL MOST LIKELY **NOT** RUN IN THE ALOTTED 9 HOURS OF RUNTIME.
    TO GET THE FULL DATA, DOWNLOAD THE SCRIPT AND RUN IT ON YOUR OWN COMPUTER.
    
    This script will analyse and create face bounding boxes for each frame of every video
    in the testing set and the sample training set. To get the full dataset, run this script
    offline. it will take several hours, depending on the power of you machine. Once I have
    it finished, I will upload it for public use. Due to technical and financial limitations,
    I will not be able to run this script on the full 500GB dataset. Hopefully someone
    else in this community will be able to help out with that :)!
    
    The data is organized in the following columns:
        - filename
        - frame
        - face: there may be multiple faces in frame at once.
        - x: x coordinate of the top left corner of the bounding box.
        - y: y coordinate of the top-left corner of the bounding box.
        - width: width of the bounding box.
        - height: height of the bounding box.
        
    I understand this script returns data that is far from perfect, but hopefully it will help
    many people get started. My personal next steps are to create a classifier based on some
    feature engineered attributes based on the regions of interest (cosine distance between ROI boxes per frame, etc.)
    
    This is a tough challenge, good luck everyone!
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt

def detect_faces_img(img, detector):
    faces = detector.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        return [[x, y, width, height] for x, y, width, height in faces]
    else:
        return [[-1, -1, -1, -1]]

def detect_faces_vid(vid, detector):
    results = list(map(lambda img: detect_faces_img(img, detector), vid))
    frames = []
    face_idx = []
    xs = []
    ys = []
    widths = []
    heights = []
    for frame, faces in enumerate(results): # might be several faces in frame!
        for idx, face in enumerate(faces):
            x, y, width, height = face
            frames.append(frame)
            face_idx.append(idx)
            xs.append(x)
            ys.append(y)
            widths.append(width)
            heights.append(height)
    df = pd.DataFrame({
        "frame": frames,
        "face": face_idx,
        "x": xs,
        "y": ys,
        "width": widths,
        "height": heights
    })
    return df

def file_to_vid(filename):
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return buf

if __name__ == "__main__":
    root = "../input"
    train = 'deepfake-detection-challenge/train_sample_videos'
    test = 'deepfake-detection-challenge/test_videos'
    train_face_df = pd.DataFrame({
        "filename": [],
        "frame": [],
        "face": [],
        "x": [],
        "y": [],
        "width": [],
        "height": []
    })
    test_face_df = pd.DataFrame({
        "filename": [],
        "frame": [],
        "face": [],
        "x": [],
        "y": [],
        "width": [],
        "height": []
    })
    face_detector = cv2.CascadeClassifier(os.path.join(root, "haarcascadefrontalfaces/haarcascade_frontalface_default.xml"))

    for dirname, _, filenames in os.walk(os.path.join(root, test)):
        for filename in filenames:
            extension = filename.split('.')[-1]
            if extension == "mp4":
                start = dt.now()
                vid = file_to_vid(os.path.join(dirname, filename))
                df = detect_faces_vid(vid, face_detector)
                df['filename'] = filename
                test_face_df = test_face_df.append(df, sort=False)
                print(filename, dt.now() - start)
    test_face_df.to_csv('test_faces.csv', index=False)

    for dirname, _, filenames in os.walk(os.path.join(root, train)):
        for filename in filenames:
            extension = filename.split('.')[-1]
            if extension == "mp4":
                start = dt.now()
                vid = file_to_vid(os.path.join(dirname, filename))
                df = detect_faces_vid(vid, face_detector)
                df['filename'] = filename
                train_face_df = train_face_df.append(df, sort=False)
                print(filename, dt.now() - start)
    train_face_df.to_csv('train_faces.csv', index=False)