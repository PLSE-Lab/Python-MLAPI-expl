import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from tqdm.notebook import tqdm_notebook
import re
from nltk.corpus import stopwords

tqdm_notebook.pandas()

# QUESTION: Why does Scenario 2 (clean text: True) perform better than Scenario 1 (clean text: false) on submission but not on validation?
#########################################################
# SCENARIO 1 VS SCENARIO 2
CLEAN_TEXT = False
#########################################################

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    # split to array(default delimiter is " ")
    text = text.split()
    text = [w for w in text if not w in set(stopwords.words('english'))]
    text = ' '.join(text)

    return text


df_train = pd.read_csv(r"../input/nlp-getting-started/train.csv")
df_test = pd.read_csv(r"../input/nlp-getting-started/test.csv")



if CLEAN_TEXT:
    df_train["text"] = df_train["text"].progress_apply(lambda x: clean_text(x))
    df_test["text"] = df_test["text"].progress_apply(lambda x: clean_text(x))

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train["text"])
X_test_tfidf = tfidf_vectorizer.transform(df_test["text"])

X_train = X_train_tfidf
y_train = df_train["target"]
X_test = X_test_tfidf

parameters = {}
lr = LogisticRegression(random_state=6, solver="lbfgs")
clf = GridSearchCV(lr, parameters, scoring="f1_micro", cv=10, n_jobs=-1)
clf.fit(X_train, y_train)

# Expected Result
print(clf.cv_results_["mean_test_score"])
#  SCENARIO 1: IF NOT CLEAN_TEXT, F1 SCORE IS: 0.72586365. SUBMISSION SCORE IS: 0.77709
#  SCENARIO 2: IF     CLEAN_TEXT, F1 SCORE IS: 0.70392749. SUBMISSION SCORE IS: 0.78118


# Predict
y_pred = clf.predict(X_test)
df_test["target"] = y_pred
df_test[["id","target"]].to_csv("submission.csv",index=False)