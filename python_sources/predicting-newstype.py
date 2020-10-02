import pandas as pd
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s

data = pd.read_csv("../input/Articles.csv",encoding="ISO-8859-1")
data = data.sample(frac=1).reset_index(drop=True) # Shuffling Rows

data["Article"] = data["Article"].str.replace("strong>","")

data['Article'] = [cleaning(s) for s in data['Article']]

sports = data[data['NewsType'] == 'sports']
business = data[data['NewsType'] == 'business']
sports_Words = pd.Series(' '.join(sports['Article'].astype(str)).lower().split(" ")).value_counts() # 20 most common words
business_Words = pd.Series(' '.join(business['Article'].astype(str)).lower().split(" ")).value_counts() # 20 most common words


vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['Article'])

encoder = LabelEncoder()
y = encoder.fit_transform(data['NewsType'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



print('<================ K Nearest Neighbours =================>')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
print('Correct % for KNN Neighbours with (K = 3) : ' + str(neigh.score(x_test, y_test)*100)+ ' %')

print('<=======================================================>\n')



print('<================== Nueral Networks ====================>')
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6,), random_state=1)
clf.fit(x_train,y_train)
print('Correct % for Nueral Networks : ' + str(clf.score(x_test, y_test)*100)+ ' %')

print('<=======================================================>\n')


print('<================= Logistic Regression =================>')
logistic = LogisticRegression()
logistic.fit(x_train,y_train)
print('Correct % for Logistic Regression : ' + str(logistic.score(x_test, y_test)*100) + ' %')

print('<=======================================================>\n')


print('<============== Naive Bayes ============================>')
gnb = MultinomialNB()
gnb.fit(x_train, y_train)
print('Correct % for Naive Bayes : ' + str(gnb.score(x_test, y_test)*100)+ ' %')

print('<========================================================>\n')