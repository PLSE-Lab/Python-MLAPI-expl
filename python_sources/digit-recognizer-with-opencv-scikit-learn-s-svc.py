import pandas
import numpy
from sklearn.model_selection import train_test_split
import sklearn.svm
import matplotlib.pyplot as pyplot
import cv2
import datetime

start_time = datetime.datetime.now()
print("start_time = {}".format(start_time))

def prepare_imgs(imgs):
    res = []
    for i in range(len(imgs)):
        current_img = imgs.iloc[i].as_matrix().reshape((28,28))
        current_img = numpy.array(current_img, dtype=numpy.uint8)

        current_img = cv2.pyrUp(current_img)
        current_img = cv2.pyrUp(current_img)
        current_img = cv2.GaussianBlur(current_img, (7,7), 0)
        current_img = cv2.pyrDown(current_img)
        current_img = cv2.pyrDown(current_img)
        
        _ ,current_img = cv2.threshold(current_img, 100, 255, cv2.THRESH_BINARY)
        current_img = current_img * 1/255

        res.append(current_img.reshape(784))
    return res

training_data  = pandas.read_csv("../input/train.csv")
training_imgs, testing_imgs, training_labels, testing_labels = train_test_split(training_data.drop("label", axis=1), training_data["label"], train_size=0.8, test_size=0.2)

# Smoothing images
final_training_imgs = prepare_imgs(training_imgs)
final_testing_imgs = prepare_imgs(testing_imgs)

# Training SVC
print("Starting to train the SVM! ({} minutes passed)".format((datetime.datetime.now()-start_time).seconds/60))
classifier = sklearn.svm.SVC()
classifier.fit(final_training_imgs, training_labels.values.ravel())
print("The classifier's score is {}".format(classifier.score(final_testing_imgs, testing_labels)))
print("{} minutes passed".format((datetime.datetime.now()-start_time).seconds/60))

# running classifier with real test data:
test_data  = pandas.read_csv("../input/test.csv")

prepared_imgs = prepare_imgs(test_data)

results = classifier.predict(prepared_imgs)

output_csv = pandas.DataFrame(data=results)
output_csv.index += 1 # So the indices starts from 1 and not 0
output_csv.to_csv("results.csv", header=["label"], index_label="ImageId")

end_time = datetime.datetime.now()

print("it took {} minutes!".format((end_time - start_time).seconds/60))
