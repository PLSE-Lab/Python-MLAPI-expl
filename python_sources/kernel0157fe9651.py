subject05.centerlight
subject05.glasses
subject05.happy
subject05.leftlight
subject05.noglasses
subject05.normal
subject05.rightlight
subject05.sad
subject05.sleepy
subject05.surprised
subject05.wink
subject01.gif
subject01.glasses
subject01.glasses.gif
subject01.happy
subject01.leftlight
subject01.noglasses
subject01.normal
subject01.rightlight
subject01.sad
subject01.sleepy
subject01.surprised
subject01.wink
cv2
os
Image 
numpy 
import cv2, os
import numpy as np
from PIL import Image
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
FaceRecognizer.train
FaceRecognizer.predict
createEigenFaceRecognizer()
createFisherFaceRecognizer(
    createLBPHFaceRecognizer()
    recognizer = cv2.createLBPHFaceRecognizer()
    def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels