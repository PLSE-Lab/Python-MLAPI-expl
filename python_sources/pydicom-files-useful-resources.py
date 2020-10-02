#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Learn about PyDICOM files.
# Below list of Kaggle kernels, resources and links from where you can learn more about DICOM format and what tools you can use to extract content from the DICOM files.
# 
# * Kevin Mader, Lung Opacity Overview, https://www.kaggle.com/kmader/lung-opacity-overview
# * Modality Specific Modules, DICOM Standard, http://dicom.nema.org/medical/dicom/2014c/output/chtml/part03/sect_C.8.html
# * DICOM Standard, https://www.dicomstandard.org/
# * Getting Started with Pydicom, https://pydicom.github.io/pydicom/stable/getting_started.html
# * ITKPYthon package, https://itkpythonpackage.readthedocs.io/en/latest/
# * DICOM in Python: Importing medical image data into NumPy with PyDICOM and VTK, * * * https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
# * DICOM Processing and Segmentation in Python, https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# * DICOM Standard Browser, https://dicom.innolitics.com/ciods
# * How can I read a DICOM image in Python, https://www.quora.com/How-can-I-read-a-DICOM-image-in-Python
# * DICOM read example in Python, https://www.programcreek.com/python/example/97517/dicom.read_file
# * DICOM in Python, https://github.com/pydicom

# In[ ]:




