import graphviz
import pandas as pd

from sklearn import metrics, preprocessing, tree  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier

# Any results you write to the current directory are saved as output.
passmark = 50

# Load dataset
dataset = pd.read_csv("../input/StudentsPerformance.csv", header=0, delimiter=',')


data_list = dataset.values.tolist()
overallScore = []

# Calculate average score for each student: math score + reading score + writing score
for i in range(len(dataset)):
    totalScore = data_list[i][5] + data_list[i][6] + data_list[i][7]
    overallScore.append(totalScore)

# Calculate average score for each student:
averageScore = [x / 3 for x in overallScore]

# Convert average score to result
for i in range(len(averageScore)):
    if averageScore[i] < passmark:
        averageScore[i] = 0
    else:
        averageScore[i] = 1

# Create new column named result as Pass(1) or Fail(0)
dataset['result'] = averageScore


# Split dataset in features and target variable
feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course',
                'math score', 'reading score', 'writing score']

# Encode string inputs
lE = preprocessing.LabelEncoder()
dataset = dataset.apply(lE.fit_transform)


X = dataset[feature_cols]  # Features
y = dataset.result  # Target variables

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini")

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

student_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True,
                                feature_names=feature_cols,
                                class_names=['FAIL', 'PASS'])

# Export the decision tree
graph = graphviz.Source(student_data)
graph.render()


#... coded by fyyavuz and myavuz.