import os
import dicom
import pandas as pd
import numpy as np

data_dir = '../input/sample_images/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)

labels_df.head()

for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
print(len(slices),label)
print (slices[0])

for patient in patients[:3]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
print(slices[0].pixel_array.shape, len(slices))

len(patients)

import matplotlib.pyplot as plt

for patient in patients[:1]:
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient

slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

plt.imshow(slices[0].pixel_array)
plt.show()
