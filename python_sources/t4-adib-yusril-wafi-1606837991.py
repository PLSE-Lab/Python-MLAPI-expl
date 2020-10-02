#!/usr/bin/env python
# coding: utf-8

# <h2><center>Tugas Individu 4 Data Mining 2019/2020</center></h2>

# 
# <center>Tanggal Launching: 5 November 2019<center>
# <center><font color="red"><b>Deadline: 14 November 2019, pukul 22.00</b></font><center>
# 
# 
# 
# 

# <h4>Nama          : Adib Yusril Wafi </h4>
# <h4>NPM           : 1606837991 </h4>
# <h4>Kolaborator   : Tidak Ada </h4>
# <h4>Kaggle   : https://www.kaggle.com/adibyw/t4-adib-yusril-wafi-1606837991 </h4>

# Petunjuk umum:<br>
# 
# 1. Dataset yang digunakan pada tugas ini merupakan dataset yang sama dengan tugas 3, yaitu dataset diabetes yang dapat diunduh di https://bit.ly/2M5vioB
# 
# 2. Lakukan pengolahan data dan perhitungan menggunakan bahasa pemrograman Python. Gunakan template Jupyter notebook yang telah disediakan untuk menjawab soal.
# 
# 3. Penggunaan Library dibatasi hanya untuk penggunaan numpy array dan pandas untuk membaca data. Khusus untuk implementasi Multilayer Perceptron, Anda dapat menggunakan library (Sklearn, Keras atau library lain).
# 
# 4. Format penulisan nama file di Jupyter notebook: T4_Nama_NIM.ipynb
# 
# 5. Kumpulkan pada slot yang disediakan di scele. Deadline: 14 November 2019, 22.00 WIB.
# 
# 6. Jika dalam menyelesaikan tugas ini anda berkolaborasi dengan orang lain, silahkan dituliskan dengan siapa anda berkolaborasi.
# 

# #### Import Library

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# #### Diabetes Dataset

# In[ ]:


data = pd.read_csv("../input/dataset/diabetes.csv")
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = ['Outcome']
print(data.shape)
data.head()


# #### Range Normalization

# In[ ]:


for column in data:
    sorted_col = data.sort_values(column)
    x_max = sorted_col[column].iloc[-1]
    x_min = sorted_col[column].iloc[0]
    range_x = x_max - x_min
    
    for i in range(data.shape[0]):
        data.loc[i, column] = (data[column][i] - x_min) / range_x
    
data.head()


# #### Neural Network

# 1.Bagilah dataset diabetes menjadi training dan testing dengan proporsi 80:20. 

# In[ ]:


training = data.loc[:(768 * 80 // 100)]
test = data.loc[(768 * 80 // 100 + 1):]

x_training, y_training = training[features].values, training[target].values
x_test, y_test = test[features].values, test[target].values
print(x_training.shape, y_training.shape)
print(x_test.shape, y_test.shape)


# Buatlah model Perceptron untuk mengklasifikasikan dataset tersebut. Gunakan <i> online learning </i> untuk meng<i>update weight </i> dan bias. Jelaskan fungsi aktivasi yang Anda gunakan dan bagaimana Anda memilih parameter yang optimal (misal <i>learning rate</i>).

# In[ ]:


# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation > 0.0 else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]    # Weights[0] is bias
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]

    return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions

l_rate = 0.1
n_epoch = 50
predicted = perceptron(training.values, test.values, l_rate, n_epoch)

for i in range(10):
    print(f'Predicted: {predicted[i]}, Actual:{y_test[i][0]}')


# #### Fungsi Aktifasi
# Fungsi aktifasi berupa jumlah dari perkalian atribut data ke-i dengan berat atribut tersebut dan kemudian ditambahkan dengan bias. Apabila hasil penjumlahan lebih dari 0 maka prediksi menghasilkan 1, sedangkan jika kurang dari atau sama dengan 0, maka prediksi akan menghasilkan 0.
# 
# #### Parameter Optimal
# Parameter yang diambil seperti learning rate diambil berdasarkan contoh pada slide di scele, sedangkan nilai epoch didapat setelah penulis bereksperimen berkali-kali untuk mencari nilai maksimal yang didapat. 

# 2.Dari hasil prediksi diatas, hitunglah akurasi, precision dan recall pada tiap kelas. Jika Anda melakukan variasi parameter, tampilkan hasil metrik evaluasi tersebut pada setiap variasi parameter yang Anda gunakan (bisa berupa tabel atau grafik).

# In[ ]:


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i][0] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

accuracy = accuracy_metric(y_test,predicted)
print(f'Accuracy = {accuracy}')


# In[ ]:


# Calculate precision percentage
def precision_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i][0] == 1 and predicted[i] == 1:
            correct += 1
    return correct / predicted.count(1) * 100.0

precision = precision_metric(y_test,predicted)
print(f'Precision = {precision}')


# In[ ]:


# Calculate recall percentage
def recall_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i][0] == 1 and predicted[i] == 1:
            correct += 1
    return correct / test[test['Outcome'] == 1].shape[0] * 100.0

recall = recall_metric(y_test,predicted)
print(f'Recall = {recall}')


# *Note: Nilai Recall dan Precision sama dikarenakan jumlah data dengan outcome = 1 pada hasil prediksi dan data test sama*

# Reference for Neural Network: 
# - https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

# #### Ensemble Learning

# 3.Dengan proporsi training dan testing yang sama dengan soal nomor 1, lakukan prediksi pada dataset diabetes dengan ketentuan sebagai berikut.
# 
# Buatlah <i> bagging classifier</i> dengan menggunakan Perceptron yang Anda buat sebagai<i> base learner </i>. Lakukan variasi terhadap jumlah <i>bootstrap sample</i>. Dari prediksi bagging perceptron, tampilkan akurasi, precision dan recall per kelas pada tiap variasi jumlah <i>bootstrap sample</i> (bisa berupa tabel atau grafik).

# In[ ]:


from random import randrange, seed

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio=1.0):
    sample = list()
    total_sample = round(len(dataset) * ratio)
    while len(sample) < total_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Make a prediction with a list of bagged data
def bagging_predict(samples, test, l_rate, n_epoch):
    predictions = [perceptron(sample, test, l_rate, n_epoch) for sample in samples]
    
    voted_predictions = list()
    
    for i in range(len(predictions[0])):
        count_0 = 0
        count_1 = 0
        
        for prediction in predictions:
            if prediction[i] == 0.0:
                count_0 += 1
            else:
                count_1 += 1
        
        if count_0 <= count_1:
            voted_predictions.append(1.0) 
        else:
            voted_predictions.append(0.0) 
    
    return voted_predictions

# Bootstrap Aggregation Algorithm
def bagging(train, test, l_rate, n_epoch, sample_size, total_samples):
    samples = list()
    
    for i in range(total_samples):
        sample = subsample(train, sample_size)
        samples.append(sample)
        
    prediction = bagging_predict(samples, test, l_rate, n_epoch)
    return prediction

def evaluate_bagging(train, test, l_rate, n_epoch, sample_size, total_samples):
    predicted = bagging(train, test, l_rate, n_epoch, sample_size, total_samples)
    
    accuracy = accuracy_metric(y_test, predicted)
    precision = precision_metric(y_test, predicted)
    recall = recall_metric(y_test, predicted)
    
    return accuracy, precision, recall

seed(1)
sample_size = 0.50
l_rate = 0.1
n_epoch = 20

for n_sample in [1, 5, 10, 50]:
    accuracy, precision, recall = evaluate_bagging(training.values, test.values, l_rate, n_epoch, sample_size, n_sample)
    
    print(f'Number of Samples: {n_sample}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision = {precision}')
    print(f'Recall = {recall}')
    print("="*50)


# Reference for Bagging Classifier: 
# - https://machinelearningmastery.com/implement-bagging-scratch-python/

# 4.Dengan proporsi training dan testing yang sama dengan soal nomor 1, lakukan prediksi pada dataset diabetes dengan ketentuan sebagai berikut.
# 
# Implementasikan <i>bagging classifier</i> yang sudah Anda buat dengan perubahan pada <i> base learner </i>. Pada eksperimen ini, gunakan multilayer perceptron (MLP) sebagai <i> base learner </i>. Anda dapat menggunakan <i>library</i> untuk mengimplementasikan MLP (Sklearn atau Keras). Anda dapat melakukan variasi terhadap jumlah bootstrap sample, jumlah hidden layer, jumlah output unit, jumlah hidden unit, atau variasi pada fungsi aktivasi. Tampilkan akurasi, precision dan recall per kelas pada tiap variasi yang Anda lakukan (bisa berupa tabel atau grafik). 

# In[ ]:


from sklearn.neural_network import MLPClassifier

# Make a prediction with a list of bagged data
def bagging_predict(samples, test):
    predictions = list()
    
    for sample in samples:
        x_training = [arr[:8] for arr in sample]
        y_training = [arr[-1] for arr in sample]
        
        clf.fit(x_training, y_training)
        predictions.append(clf.predict(x_test))
    
    voted_predictions = list()
    
    for i in range(len(predictions[0])):
        count_0 = 0
        count_1 = 0
        
        for prediction in predictions:
            if prediction[i] == 0.0:
                count_0 += 1
            else:
                count_1 += 1
        
        if count_0 <= count_1:
            voted_predictions.append(1.0) 
        else:
            voted_predictions.append(0.0) 
    
    return voted_predictions

# Bootstrap Aggregation Algorithm
def bagging(train, test, sample_size, total_samples):
    samples = list()
    
    for i in range(total_samples):
        sample = subsample(train, sample_size)
        samples.append(sample)
        
    prediction = bagging_predict(samples, test)
    return prediction

def evaluate_bagging(train, test, sample_size, total_samples):
    predicted = bagging(train, test, sample_size, total_samples)
    
    accuracy = accuracy_metric(y_test, predicted)
    precision = precision_metric(y_test, predicted)
    recall = recall_metric(y_test, predicted)
    
    return accuracy, precision, recall

seed(2)
sample_size = 0.50

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,2), random_state=2)


for n_sample in [1, 5, 10, 50, 70]:
    accuracy, precision, recall = evaluate_bagging(training.values, test.values, sample_size, n_sample)
    
    print(f'Number of Samples: {n_sample}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision = {precision}')
    print(f'Recall = {recall}')
    print("="*50)


# Reference for Multi Layer Perceptron: 
# - https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# 5.Lakukan komparasi dan analisis dari keseluruhan hasil eksperimen yang Anda lakukan (Perceptron, BaggingPperceptron, Bagging MLP) untuk menyelesaikan permasalahan klasifikasi penyakit diabetes. 

# #### Analysis
# The three algorithm above have `similar values` in terms of accuracy in classifying the diabetes data set (hovering around `70-80%` accuracy), and the precision and recall values of each have varying values depending on the condition the algorithm was run on.
# 
# #### Perceptron
# The perceptron algorithm gave us an accuracy score of 77.8%, while the precision and recall value both gave us a 68% score. This result will vary depending on the starting `weights` and `bias` we set on perceptron and number of `epoch` we allow the algorithm to train the `weights` and the `bias`.
# 
# I found out that if we set the `learning rate` of the percepton to be a big number (>= 1) it would create a big gap on each iteration where we train the `weights` and `bias` of our algorithm. If we use a bigger `epoch` number it will allow our algorithm to reach a more optimum `weights` for the perceptron, but if we use a number that is more than necessary, then our algorithm will have a bloated weights that could prove to decrease the accuracy of the prediction.
# 
# #### Bagging Perceptron
# This algorithm uses the same perceptron as the previous algorithm and the things that differentiate this algorithm is the number of training samples and the size of each samples. With enough experiment, I found that if we use more samples to predict the result, it will give us a better accuracy, but if we set the size of each sample to be very small or very big, it will not really help increasing the accuracy of our algorithm because of the bias and variance contained in our sample 
# 
# #### Bagging MLP
# In this algorithm, the parameter that could determines the accuracy of our prediction is the `alpha` and `hidden layer` in our `multi layer perceptron`. After experimenting with changin the values of the `hidden layer` and `alpha` of the `multi layer perceptron`, I found some combination gives better accuracy on smaller number of samples while other combination give a irregular result depending on the sample sizes. In the end, I choose a combination that has a consistency in which the larger the number of samples we used, the better accuracy we will get.
