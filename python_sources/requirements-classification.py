import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xlsxwriter
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC  # "Support Vector Classifier"
from sklearn import svm
from sklearn.metrics import f1_score
class DataProcessor:
    def __init__(self, specific=False, vectorization='USE'):
        self.train, self.test = get_data(specific=specific)
        self.le = LabelEncoder()
        self.le.fit(self.train[1])                              #train[0]: example and train[1]: label
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(self.train[0])

        if vectorization == 'USE':
            self.vectorization = use_embeddings
        elif vectorization == 'ELMo':
            self.vectorization = elmo_embeddings
        else:
            self.vectorization = self.tfidf_embeddings

        self.train = (self.vectorization(self.train[0]), self.le.transform(self.train[1]))
        self.test = (self.vectorization(self.test[0]), self.le.transform(self.test[1]))

    def transform(self, label):
        return self.le.transform(label)

    def inverse_transform(self, label):
        return self.le.inverse_transform(label)

    def tfidf_embeddings(self, examples):

        return self.tfidf_vectorizer.transform(examples).todense()


def use_embeddings(examples):
    embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3')
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(examples))


    return embeddings

def elmo_embeddings(examples, batch_size=512):


    return 0






def process_line(line, specific=False):
    parts = line.split()
    label = parts[0]
    if not specific:
        label = label.split(':')[0]

    example = ' '.join(parts[1:])

    return label, example






def read_file(filepath, specific= False):
    x, y = [], []
    with open(filepath, encoding="ISO-8859-1") as in_file:
        line = in_file.readline()
        while line:
            Y_, X_ = process_line(line, specific = specific)            # Y_ contains label, X_ contains example
            x.append(X_)
            y.append(Y_)
            # print(X_)
            # print(Y_)
            line = in_file.readline()

    workbook = xlsxwriter.Workbook('Example1.xlsx')
    worksheet = workbook.add_worksheet()

        # Start from the first cell.
        # Rows and columns are zero indexed.
    row = 0
    column1 = 0
    for item in x:
        #print(item)
        worksheet.write(row, column1, item)
        row += 1

    workbook.close()


    return x, y                                                         # x contains example, y contains label




def get_data(train_path='../input/requirementsclassification/nfr.txt', test_path='../input/requirementsclassification/test.txt', specific=False):
    train_x, train_y = read_file(train_path, specific=specific)
    test_x, test_y = read_file(test_path, specific=specific)
    return (train_x, train_y), (test_x, test_y)                         #train_x contains example, train_y contains label



def caller():
    data = DataProcessor(vectorization='TFIDF')
    return data
#     import pdb
#     pdb.set_trace()

def main():
    #data = DataProcessor(vectorization='TFIDF')
    data = caller()
    train_x, train_y = data.train
    test_x, test_y = data.test
    clf = SGDClassifier(loss='log')
    #clf = LogisticRegression()
    #clf = RidgeClassifier()
    #clf = PassiveAggressiveClassifier(loss='log')
    #clf = RidgeClassifierCV()
    #clf = SVC(kernel='linear')
    #clf = svm.SVC(gamma=0.01, C=5)
    # fitting x samples and y classes

    clf.fit(train_x, train_y)
    #p = clf.predict('What is the temperature of sun')
    train_pred = clf.predict(train_x)
    test_pred = clf.predict(test_x)
    test_label = data.inverse_transform(test_pred)
    test_ori_label = data.inverse_transform(test_y)

    #print(test_y)
    #print(test_pred)
   #  print(test_ori_label)
   #  print(test_label)

    workbook = xlsxwriter.Workbook('Example2.xlsx')
    worksheet = workbook.add_worksheet()

    # Start from the first cell.
    # Rows and columns are zero indexed.
    row = 0
    column1 = 0
    column2 = 1
    column3 = 2


    for item in test_ori_label:
        worksheet.write(row, column2, item)
        row += 1
    row = 0
    for item in test_label:
        worksheet.write(row, column3, item)
        row +=1

    workbook.close()



    print('Training performance')
    print(f1_score(train_y, train_pred, average='weighted'))
    print('Test performance')
    print(f1_score(test_y, test_pred, average='weighted'))


if __name__ == '__main__':
    main()




