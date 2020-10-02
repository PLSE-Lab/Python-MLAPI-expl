# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import csv
import time
from sklearn import metrics
from sklearn.metrics import average_precision_score, confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.

def f1_Score_Function(raw_data):
    
    raw_data = raw_data[raw_data.index("support"):-2]
    raw_data=raw_data.split("      ")   
    precision = raw_data[2]
    recall = raw_data[3]
    f1_score = raw_data[4]
    precision = precision[0:6]
    recall = recall[0:4]
    f1_score = f1_score[0:4]
    
    return precision, recall, f1_score

repetition = 10

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

csv_files = ["Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", "Heartbleed", "Infiltration", "PortScan", "SSH-Patator", "Web Attack"]# CSV files names: #The names of the files in the attacks folder are taken and assigned to a list (csv_files).

features = {"Bot": ["Bwd Packet Length Mean", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Std", "Flow Duration", "Label"], 
            "DDoS": ["Bwd Packet Length Std", "Total Backward Packets", "Fwd IAT Total", "Flow Duration", "Flow IAT Min", "Label"],
            "DoS GoldenEye": ["Flow IAT Max", "Flow IAT Min", "Total Backward Packets", "Bwd Packet Length Std", "Bwd Packet Length Mean", "Label"],
            "DoS Hulk": ["Bwd Packet Length Std", "Fwd Packet Length Std", "Fwd Packet Length Max", "Flow Duration", "Flow IAT Min", "Label"],
            "DoS Slowhttptest": ["Flow IAT Mean", "Fwd Packet Length Min", "Fwd Packet Length Std", "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Label"],
            "DoS slowloris": ["Flow IAT Mean", "Bwd Packet Length Mean", "Total Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s", "Label"],
            "FTP-Patator": ["Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Std", "Total Length of Bwd Packets", "Label"],
            "Heartbleed": ["Bwd Packet Length Max", "Bwd Packet Length Mean", "Total Backward Packets", "Total Length of Bwd Packets", "Total Length of Fwd Packets", "Label"],
            "Infiltration": ["Total Length of Fwd Packets", "Fwd Packet Length Mean", "Flow Duration", "Flow IAT Max", "Fwd Packet Length Max", "Label"],
            "PortScan": ["Flow Bytes/s", "Total Length of Fwd Packets", "Fwd IAT Total", "Fwd Packet Length Max", "Flow Duration", "Label"],
            "SSH-Patator": ["Fwd Packet Length Max", "Flow IAT Mean", "Flow IAT Max", "Flow Duration", "Flow Bytes/s", "Label"],
            "Web Attack": ["Flow Duration", "Total Fwd Packets", "Flow IAT Min", "Flow IAT Max", "Flow IAT Std", "Label"]}

randomForest = RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1)


with open("results.csv", "w", newline = "", encoding = "utf-8") as file:
    write = csv.writer(file)
    write.writerow(["Attack File", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])
    

for j in csv_files:
    seconds = time.time() # program starting time stamp
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    time_count = []
    
    relevant_features = list(features[j])
    df = pd.read_csv("../input/attack-datasets/" + j + ".csv", usecols = relevant_features)
    df = df.fillna(0)
    
    is_attack = []
    
    for i in df["Label"]:
        
        if i == "BENIGN":
            is_attack.append(1)
            
        else:
            is_attack.append(0)
        
    df["Label"] = is_attack
    
    y = df["Label"]
    del df["Label"]
    relevant_features.remove("Label")
    X = df[relevant_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = repetition)
    
    randomForest.fit(X_train, y_train)
    
    prediction = randomForest.predict(X_test)

    prec, rec, f_1 = f1_Score_Function(metrics.classification_report(y_test, prediction))
    
    precision.append(float(prec))
    recall.append(float(rec))
    f1_score.append(float(f_1))
    
    accuracy.append(randomForest.score(X_test, y_test))
    time_count.append(float((time.time() - seconds)))
    
    with open("results.csv", "a", newline = "", encoding = "utf-8") as file:
        
        write = csv.writer(file)
        
        for i in range(0, len(time_count)):
            write.writerow([j, accuracy[i], precision[i], recall[i], f1_score[i], time_count[i]])
            print(j, accuracy[i], precision[i], recall[i], f1_score[i], time_count[i])
                    
print('Finished.')