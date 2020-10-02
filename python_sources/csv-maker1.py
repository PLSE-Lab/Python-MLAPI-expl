# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import csv
import sys
import os
#Useful function
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
folders = os.listdir('/kaggle/input/hand-sign-recognition/original_images/')
folders.sort()
print(folders)

name=folders
print(name)
len(name)

myFileList = createFileList('/kaggle/input/hand-sign-recognition/original_images/'+str(name[0])+'/')

for n in name:
    #print('/kaggle/input/hand-sign-recognition/original_images/'+str(n)+'/')
    myFileList = createFileList('/kaggle/input/hand-sign-recognition/original_images/'+str(n)+'/')
    
    for file in myFileList:
        #x = input(str(n)+" completed. Continue? (y/n)")
        #if x == 'n':
            #break
        break  #remove this break statement to start the process as it requires enourmous space
        print(file)
        img_file = Image.open(file)
        # img_file.show()
        # get original image parameters...
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        # Make image Greyscale
        img_grey = img_file.convert('L')
        #img_grey.save('result.png')
        #img_grey.show()

        # Save Greyscale values
        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        value = np.append(value,n)
        print(value)
        with open("img_pixels.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
        #x = input(str(n)+" completed. Continue? (y/n)")
        #if x == 'n':
            #break
