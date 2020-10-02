
import pandas as pd
import numpy as np
from PIL import Image
import os

# Load data
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv("../input/test.csv")
    
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