import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

with open('../input/train.csv', 'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    feature = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])
        feature.append(data[3:])
print('Loaded ' + str(len(train_label)))

print('Read testing data...')
with open('../input/test.csv', 'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int, line.rstrip().split(',')))
        test_data.append(pixels)
print('Loaded ' + str(len(test_data)))


feature_train, feature_test, target_train, target_test = train_test_split(feature, train_label, test_size=0.1, random_state=42)
clf = RandomForestClassifier(n_estimators=10)
s = clf.fit(train_data , train_label)

predict_result_list = s.predict(test_data)
print(type(predict_result_list))
with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict_result_list:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')
