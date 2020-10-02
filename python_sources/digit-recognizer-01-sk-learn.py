# Import statements 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt #needed?
import time
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model
import warnings
warnings.filterwarnings("ignore")
# Data I/O
train_data = pd.read_csv('../input/train.csv')
train_data.shape #(42000, 785)
# train_data.head()
# Data Preprocessing
labels = train_data['label']
images = train_data.drop('label', axis=1)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
train_images_binary = train_images.copy(deep=True)
test_images_binary = test_images.copy(deep=True)
train_images_binary[train_images_binary>0] = 1
test_images_binary[test_images_binary>0] = 1
images_binary = images.copy(deep=True)
images_binary[images_binary>0] = 1
# Model Fitting 
all_models = {'svm': svm.SVC(gamma='auto', max_iter=750, random_state=0), #800s
              'linear-svm': svm.LinearSVC(max_iter=1750, random_state=0), #100s
              'linear': linear_model.ElasticNet(max_iter=1000, random_state=0), #6s
              'logistic': linear_model.LogisticRegressionCV(max_iter=50, n_jobs=-1, random_state=0), #680s
              'pac': linear_model.PassiveAggressiveClassifier(random_state=0)} #5s
grayscale_results = pd.DataFrame(columns=all_models.keys(), index=['train', 'test'])
binary_results = pd.DataFrame(columns=all_models.keys(), index=['train', 'test'])
print('Starting Training')
for model_type, model in all_models.items(): 
    print(f"Fitting {model_type} model...")
    start = time.time()
    # model.fit(train_images, train_labels)
    # grayscale_results.loc['train', model_type] = model.score(train_images, train_labels)
    # grayscale_results.loc['test', model_type] = model.score(test_images, test_labels)
    model.fit(train_images_binary, train_labels)
    binary_results.loc['train', model_type] = model.score(train_images_binary, train_labels)
    binary_results.loc['test', model_type] = model.score(test_images_binary, test_labels)
    print(f"Total time to fit and score `{model_type}` model: {round(time.time()-start,2)} seconds")
print('Training Done!')
# Model Selection
grayscale_results
# best grayscale: logistic, pac, linear-svm
#       svm       linear_svm  linear   logistic    pac
#train  0.385119   0.864435  0.614685  0.932143  0.888065
#test   0.206429   0.844524  0.605862  0.909881  0.875952
binary_results
# best binary: svm, logistic, linear-svm
#         svm     linear-svm  linear     logistic    pac
#train   0.94497   0.932054  0.0374634   0.92253   0.875387
#test    0.942857  0.9025    0.038317    0.909167  0.861429
#Overall best model: Binary SVM
model = svm.SVC(gamma='auto', random_state=0)
model.fit(images_binary, labels) #train_images_binary, train_labels
print('Finished fitting final model.')
model.score(test_images_binary, test_labels)
# Test Data Results
test_data = pd.read_csv('../input/test.csv')
test_data_binary = test_data.copy(deep=True)
test_data_binary[test_data_binary>0] = 1
results_raw = model.predict(test_data_binary) #test_data / test_data_binary
results = pd.DataFrame(results_raw)
results.columns = ['Label']
results.index.name = 'ImageId'
results.index += 1
results.to_csv('results.csv', header=True)














