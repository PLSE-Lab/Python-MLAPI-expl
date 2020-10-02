import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

def load_data(file):
    labeled_images = pd.read_csv(file)
    images = [[x > 0 for x in row] for row in labeled_images.iloc[0:5000, 1:].values]
    labels = labeled_images.iloc[0:5000, :1].values.ravel()
    return images, labels
    
images, labels = load_data("../input/train.csv")
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

# print(len(train_images), len(test_images), len(train_images[0]))

clf = svm.SVC()
clf.fit(train_images, train_labels)
score = clf.score(test_images, test_labels)

# print(score)

test_data = pd.read_csv("../input/test.csv")
test_data[test_data > 0] = 1
prediction = clf.predict(test_data[0:5000])

df = pd.DataFrame(prediction)
df.index.names = ["ImageId"]
df.columns = ["Label"]
df.to_csv("prediction.csv", header=True)
