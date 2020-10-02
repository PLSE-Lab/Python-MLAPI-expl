#Utilities I often use

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 

def confussion_pies(y_test, y_pred):
    labels = y_test.unique()
    labels.sort()
    count_labels = len(labels)
    fig, axs = plt.subplots(ncols=int(count_labels), figsize=(20,10))
    col = 0
    row = 0
    for label in labels:
        axs[row].set_title('Class: {}'.format(label))
        predicted = np.where(y_pred == label )[0].size
        total = np.where( y_test == label )[0].size
        failed = total - predicted
        message = 'above' if failed < 0 else 'under'
        
        labels = ['Predicted correctly', 'Prediction Failed {} test count'.format(message)]
        sizes = [predicted, failed if failed > 0 else abs(failed)]
        colors = ['green', 'red']
        explode = (0.1, 0)  # explode 1st slice

        # Plot
        axs[row].pie(sizes, labels = labels,explode = explode, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)

        axs[row].axis('equal')
        row = row + 1
    plt.show()