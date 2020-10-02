import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 2000)
train_path = '../input/train.csv'
test_path = '../input/test.csv'
sub_path = 'submission.csv'
def clean_text(text):
    return text

#Create training and Testing Data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
df1 = train_df[train_df.target == 1]
df0 = train_df[train_df.target == 0].sample(len(df1)*6)
df = pd.concat([df0,df1])
train_df = df.sample(frac=1).reset_index(drop=True)
vectorizer = CountVectorizer(stop_words="english", preprocessor=clean_text)
training_features = vectorizer.fit_transform(train_df["question_text"])
test_features = vectorizer.transform(test_df["question_text"])

# ################   SVC   #######################
print("Building SVC")
SVC = LinearSVC(max_iter = 5000, dual = False)
SVC.fit(training_features, train_df['target'])
SVC_pred = SVC.predict(test_features)
test_df['SVC_prediction'] = SVC_pred

# ################   LR   #######################
print("Building Logistic Regression")
LR = LogisticRegression()
LR.fit(training_features, train_df['target'])
LR_pred = LR.predict(test_features)
test_df['LR_prediction'] = LR_pred

# ######### SGD Classifier ####################
print("Building SGD")
SGD = SGDClassifier(max_iter = 5000)
SGD.fit(training_features, train_df['target'])
SGD_pred = SGD.predict(test_features)
test_df['SGD_prediction'] = SGD_pred


# ######### Ensembling ####################
print("Building Ensemble 1")
test_df['sum'] = test_df['LR_prediction'] + test_df['SGD_prediction'] + test_df['SVC_prediction']
test_df['prediction'] = (test_df['sum'] >= 2).astype('int')

submission_df = test_df[['qid','prediction']]
submission_df.to_csv(sub_path,index = False)

