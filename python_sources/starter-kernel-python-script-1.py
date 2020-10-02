# This starter kernel is a basic fork of https://www.kaggle.com/joseduc/digit-recognizer/training-data-to-png/code
# Run it to print a small sample of the training data.

import pandas as pd
import numpy as np
from PIL import Image
import os

# Load data
train = pd.read_csv('../input/train.csv')
    
# Draw a sample of the numbers
for ind, row in train.iloc[1:10].iterrows():
    i = row['label']
    # uncomment the line below if you prefer black numbers
    # over a white background
    #arr = np.array(255 - row[1:], dtype=np.uint8)
    arr = np.array(row[1:], dtype=np.uint8)
    arr.resize((28, 28))
    im = Image.fromarray(arr)
    im.save("%s-%s.png" % (i, ind))