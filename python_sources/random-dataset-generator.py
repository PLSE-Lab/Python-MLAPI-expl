# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

 # linear algebra
 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory




# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
print("hello")

# import tensorflow as tf
# import cv2
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
#reference https://python-graph-gallery.com/122-multiple-lines-chart/
dirName = 'data_set'
 
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

def random_multi_line_chart():
    for i in range(1, 400):
        print("hello")
        list_y = [];
        list_y = [np.random.randint(200, 5000) for i in range(1, 100)]
        marker_size = np.random.randint(4, 12)
        abc = list_y;
        plt.plot()
        plt.plot(abc,  color='skyblue', linewidth=1)

        colors = ['aqua', 'black', 'blue', 'fuchsia', 'gray', 'green',
                  'lime', 'maroon', 'navy', 'olive', 'orange', 'purple', 'red',
                  'silver', 'teal', 'white', 'yellow'];
        for i in range(np.random.randint(1,3)):
             list_y_2 = [np.random.randint(200, 5000) for i in range(1, 100)]
             plt.plot(list_y_2, color=colors[np.random.randint(1,15)],
                      linewidth=1)

        # plt.legend()

        plt.savefig("data_set/multiline_" + str(i) + ".jpg")
        plt.show()



random_multi_line_chart();
