import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
# The competition datafiles are in the directory ../input
# Read competition data files
data_file = pd.read_csv("../input/train.csv")

label = data_file['label']
pixels = data_file.ix[:,1:]


##reshaping the arrays 
##convert arrays,specify data type and reshape 

label = label.astype(np.uint8)
pixels = np.array(pixels).reshape((-1, 1, 28, 28)).astype(np.uint8)

plt.imshow(pixels[1729][0], cmap=cm.binary)
##plt.imshow(pixels[0][0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show