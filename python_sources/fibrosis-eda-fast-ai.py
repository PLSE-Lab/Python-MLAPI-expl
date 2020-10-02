#!/usr/bin/env python
# coding: utf-8

# # Pulmonary Fibrosis EDA
# 
# Another image competition goes by, and this is rather interesting. I will be interested to see how well people can create their own solutions or apply solutions from Pneumothorax Segmentation competition or, if you really want to go down memory lane, perhaps Data Science Bowl 2017.

# # Insights
# 
# + **A.** There are a variable number of slices per patient, ranging from 20-30 to 400.
# + **B.** Not much relationship between the principal image features per scan (for certain images)

# In[ ]:


# FROM https://gist.github.com/FedeMiorelli/640bbc66b2038a14802729e609abfe89
import numpy as np
import matplotlib.pyplot as plt

turbo_colormap_data = np.array(
                       [[0.18995,0.07176,0.23217],
                       [0.19483,0.08339,0.26149],
                       [0.19956,0.09498,0.29024],
                       [0.20415,0.10652,0.31844],
                       [0.20860,0.11802,0.34607],
                       [0.21291,0.12947,0.37314],
                       [0.21708,0.14087,0.39964],
                       [0.22111,0.15223,0.42558],
                       [0.22500,0.16354,0.45096],
                       [0.22875,0.17481,0.47578],
                       [0.23236,0.18603,0.50004],
                       [0.23582,0.19720,0.52373],
                       [0.23915,0.20833,0.54686],
                       [0.24234,0.21941,0.56942],
                       [0.24539,0.23044,0.59142],
                       [0.24830,0.24143,0.61286],
                       [0.25107,0.25237,0.63374],
                       [0.25369,0.26327,0.65406],
                       [0.25618,0.27412,0.67381],
                       [0.25853,0.28492,0.69300],
                       [0.26074,0.29568,0.71162],
                       [0.26280,0.30639,0.72968],
                       [0.26473,0.31706,0.74718],
                       [0.26652,0.32768,0.76412],
                       [0.26816,0.33825,0.78050],
                       [0.26967,0.34878,0.79631],
                       [0.27103,0.35926,0.81156],
                       [0.27226,0.36970,0.82624],
                       [0.27334,0.38008,0.84037],
                       [0.27429,0.39043,0.85393],
                       [0.27509,0.40072,0.86692],
                       [0.27576,0.41097,0.87936],
                       [0.27628,0.42118,0.89123],
                       [0.27667,0.43134,0.90254],
                       [0.27691,0.44145,0.91328],
                       [0.27701,0.45152,0.92347],
                       [0.27698,0.46153,0.93309],
                       [0.27680,0.47151,0.94214],
                       [0.27648,0.48144,0.95064],
                       [0.27603,0.49132,0.95857],
                       [0.27543,0.50115,0.96594],
                       [0.27469,0.51094,0.97275],
                       [0.27381,0.52069,0.97899],
                       [0.27273,0.53040,0.98461],
                       [0.27106,0.54015,0.98930],
                       [0.26878,0.54995,0.99303],
                       [0.26592,0.55979,0.99583],
                       [0.26252,0.56967,0.99773],
                       [0.25862,0.57958,0.99876],
                       [0.25425,0.58950,0.99896],
                       [0.24946,0.59943,0.99835],
                       [0.24427,0.60937,0.99697],
                       [0.23874,0.61931,0.99485],
                       [0.23288,0.62923,0.99202],
                       [0.22676,0.63913,0.98851],
                       [0.22039,0.64901,0.98436],
                       [0.21382,0.65886,0.97959],
                       [0.20708,0.66866,0.97423],
                       [0.20021,0.67842,0.96833],
                       [0.19326,0.68812,0.96190],
                       [0.18625,0.69775,0.95498],
                       [0.17923,0.70732,0.94761],
                       [0.17223,0.71680,0.93981],
                       [0.16529,0.72620,0.93161],
                       [0.15844,0.73551,0.92305],
                       [0.15173,0.74472,0.91416],
                       [0.14519,0.75381,0.90496],
                       [0.13886,0.76279,0.89550],
                       [0.13278,0.77165,0.88580],
                       [0.12698,0.78037,0.87590],
                       [0.12151,0.78896,0.86581],
                       [0.11639,0.79740,0.85559],
                       [0.11167,0.80569,0.84525],
                       [0.10738,0.81381,0.83484],
                       [0.10357,0.82177,0.82437],
                       [0.10026,0.82955,0.81389],
                       [0.09750,0.83714,0.80342],
                       [0.09532,0.84455,0.79299],
                       [0.09377,0.85175,0.78264],
                       [0.09287,0.85875,0.77240],
                       [0.09267,0.86554,0.76230],
                       [0.09320,0.87211,0.75237],
                       [0.09451,0.87844,0.74265],
                       [0.09662,0.88454,0.73316],
                       [0.09958,0.89040,0.72393],
                       [0.10342,0.89600,0.71500],
                       [0.10815,0.90142,0.70599],
                       [0.11374,0.90673,0.69651],
                       [0.12014,0.91193,0.68660],
                       [0.12733,0.91701,0.67627],
                       [0.13526,0.92197,0.66556],
                       [0.14391,0.92680,0.65448],
                       [0.15323,0.93151,0.64308],
                       [0.16319,0.93609,0.63137],
                       [0.17377,0.94053,0.61938],
                       [0.18491,0.94484,0.60713],
                       [0.19659,0.94901,0.59466],
                       [0.20877,0.95304,0.58199],
                       [0.22142,0.95692,0.56914],
                       [0.23449,0.96065,0.55614],
                       [0.24797,0.96423,0.54303],
                       [0.26180,0.96765,0.52981],
                       [0.27597,0.97092,0.51653],
                       [0.29042,0.97403,0.50321],
                       [0.30513,0.97697,0.48987],
                       [0.32006,0.97974,0.47654],
                       [0.33517,0.98234,0.46325],
                       [0.35043,0.98477,0.45002],
                       [0.36581,0.98702,0.43688],
                       [0.38127,0.98909,0.42386],
                       [0.39678,0.99098,0.41098],
                       [0.41229,0.99268,0.39826],
                       [0.42778,0.99419,0.38575],
                       [0.44321,0.99551,0.37345],
                       [0.45854,0.99663,0.36140],
                       [0.47375,0.99755,0.34963],
                       [0.48879,0.99828,0.33816],
                       [0.50362,0.99879,0.32701],
                       [0.51822,0.99910,0.31622],
                       [0.53255,0.99919,0.30581],
                       [0.54658,0.99907,0.29581],
                       [0.56026,0.99873,0.28623],
                       [0.57357,0.99817,0.27712],
                       [0.58646,0.99739,0.26849],
                       [0.59891,0.99638,0.26038],
                       [0.61088,0.99514,0.25280],
                       [0.62233,0.99366,0.24579],
                       [0.63323,0.99195,0.23937],
                       [0.64362,0.98999,0.23356],
                       [0.65394,0.98775,0.22835],
                       [0.66428,0.98524,0.22370],
                       [0.67462,0.98246,0.21960],
                       [0.68494,0.97941,0.21602],
                       [0.69525,0.97610,0.21294],
                       [0.70553,0.97255,0.21032],
                       [0.71577,0.96875,0.20815],
                       [0.72596,0.96470,0.20640],
                       [0.73610,0.96043,0.20504],
                       [0.74617,0.95593,0.20406],
                       [0.75617,0.95121,0.20343],
                       [0.76608,0.94627,0.20311],
                       [0.77591,0.94113,0.20310],
                       [0.78563,0.93579,0.20336],
                       [0.79524,0.93025,0.20386],
                       [0.80473,0.92452,0.20459],
                       [0.81410,0.91861,0.20552],
                       [0.82333,0.91253,0.20663],
                       [0.83241,0.90627,0.20788],
                       [0.84133,0.89986,0.20926],
                       [0.85010,0.89328,0.21074],
                       [0.85868,0.88655,0.21230],
                       [0.86709,0.87968,0.21391],
                       [0.87530,0.87267,0.21555],
                       [0.88331,0.86553,0.21719],
                       [0.89112,0.85826,0.21880],
                       [0.89870,0.85087,0.22038],
                       [0.90605,0.84337,0.22188],
                       [0.91317,0.83576,0.22328],
                       [0.92004,0.82806,0.22456],
                       [0.92666,0.82025,0.22570],
                       [0.93301,0.81236,0.22667],
                       [0.93909,0.80439,0.22744],
                       [0.94489,0.79634,0.22800],
                       [0.95039,0.78823,0.22831],
                       [0.95560,0.78005,0.22836],
                       [0.96049,0.77181,0.22811],
                       [0.96507,0.76352,0.22754],
                       [0.96931,0.75519,0.22663],
                       [0.97323,0.74682,0.22536],
                       [0.97679,0.73842,0.22369],
                       [0.98000,0.73000,0.22161],
                       [0.98289,0.72140,0.21918],
                       [0.98549,0.71250,0.21650],
                       [0.98781,0.70330,0.21358],
                       [0.98986,0.69382,0.21043],
                       [0.99163,0.68408,0.20706],
                       [0.99314,0.67408,0.20348],
                       [0.99438,0.66386,0.19971],
                       [0.99535,0.65341,0.19577],
                       [0.99607,0.64277,0.19165],
                       [0.99654,0.63193,0.18738],
                       [0.99675,0.62093,0.18297],
                       [0.99672,0.60977,0.17842],
                       [0.99644,0.59846,0.17376],
                       [0.99593,0.58703,0.16899],
                       [0.99517,0.57549,0.16412],
                       [0.99419,0.56386,0.15918],
                       [0.99297,0.55214,0.15417],
                       [0.99153,0.54036,0.14910],
                       [0.98987,0.52854,0.14398],
                       [0.98799,0.51667,0.13883],
                       [0.98590,0.50479,0.13367],
                       [0.98360,0.49291,0.12849],
                       [0.98108,0.48104,0.12332],
                       [0.97837,0.46920,0.11817],
                       [0.97545,0.45740,0.11305],
                       [0.97234,0.44565,0.10797],
                       [0.96904,0.43399,0.10294],
                       [0.96555,0.42241,0.09798],
                       [0.96187,0.41093,0.09310],
                       [0.95801,0.39958,0.08831],
                       [0.95398,0.38836,0.08362],
                       [0.94977,0.37729,0.07905],
                       [0.94538,0.36638,0.07461],
                       [0.94084,0.35566,0.07031],
                       [0.93612,0.34513,0.06616],
                       [0.93125,0.33482,0.06218],
                       [0.92623,0.32473,0.05837],
                       [0.92105,0.31489,0.05475],
                       [0.91572,0.30530,0.05134],
                       [0.91024,0.29599,0.04814],
                       [0.90463,0.28696,0.04516],
                       [0.89888,0.27824,0.04243],
                       [0.89298,0.26981,0.03993],
                       [0.88691,0.26152,0.03753],
                       [0.88066,0.25334,0.03521],
                       [0.87422,0.24526,0.03297],
                       [0.86760,0.23730,0.03082],
                       [0.86079,0.22945,0.02875],
                       [0.85380,0.22170,0.02677],
                       [0.84662,0.21407,0.02487],
                       [0.83926,0.20654,0.02305],
                       [0.83172,0.19912,0.02131],
                       [0.82399,0.19182,0.01966],
                       [0.81608,0.18462,0.01809],
                       [0.80799,0.17753,0.01660],
                       [0.79971,0.17055,0.01520],
                       [0.79125,0.16368,0.01387],
                       [0.78260,0.15693,0.01264],
                       [0.77377,0.15028,0.01148],
                       [0.76476,0.14374,0.01041],
                       [0.75556,0.13731,0.00942],
                       [0.74617,0.13098,0.00851],
                       [0.73661,0.12477,0.00769],
                       [0.72686,0.11867,0.00695],
                       [0.71692,0.11268,0.00629],
                       [0.70680,0.10680,0.00571],
                       [0.69650,0.10102,0.00522],
                       [0.68602,0.09536,0.00481],
                       [0.67535,0.08980,0.00449],
                       [0.66449,0.08436,0.00424],
                       [0.65345,0.07902,0.00408],
                       [0.64223,0.07380,0.00401],
                       [0.63082,0.06868,0.00401],
                       [0.61923,0.06367,0.00410],
                       [0.60746,0.05878,0.00427],
                       [0.59550,0.05399,0.00453],
                       [0.58336,0.04931,0.00486],
                       [0.57103,0.04474,0.00529],
                       [0.55852,0.04028,0.00579],
                       [0.54583,0.03593,0.00638],
                       [0.53295,0.03169,0.00705],
                       [0.51989,0.02756,0.00780],
                       [0.50664,0.02354,0.00863],
                       [0.49321,0.01963,0.00955],
                       [0.47960,0.01583,0.01055]])




def RGBToPyCmap(rgbdata):
    nsteps = rgbdata.shape[0]
    stepaxis = np.linspace(0, 1, nsteps)

    rdata=[]; gdata=[]; bdata=[]
    for istep in range(nsteps):
        r = rgbdata[istep,0]
        g = rgbdata[istep,1]
        b = rgbdata[istep,2]
        rdata.append((stepaxis[istep], r, r))
        gdata.append((stepaxis[istep], g, g))
        bdata.append((stepaxis[istep], b, b))

    mpl_data = {'red':   rdata,
                 'green': gdata,
                 'blue':  bdata}

    return mpl_data


mpl_data = RGBToPyCmap(turbo_colormap_data)
plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])

mpl_data_r = RGBToPyCmap(turbo_colormap_data[::-1,:])
plt.register_cmap(name='turbo_r', data=mpl_data_r, lut=turbo_colormap_data.shape[0])


# In[ ]:


get_ipython().system('pip install fastai2 -q')
get_ipython().system('pip install efficientnet_pytorch -q')
get_ipython().system('conda install -c conda-forge gdcm -y')


# In[ ]:


from fastai2.basics           import *
from fastai2.medical.imaging  import *


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dcm
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
p = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import cv2


# In[ ]:


patient_sizes = [len(os.listdir('../input/osic-pulmonary-fibrosis-progression/train/' + d)) for d in os.listdir('../input/osic-pulmonary-fibrosis-progression/train/')]
plt.hist(patient_sizes, color=p[2])
plt.ylabel('Number of patients')
plt.xlabel('DICOM files')
plt.title('Histogram of DICOM count per patient');


# As we can see here, we have a lot of patients (subdirectories) with  <100 counts being the norm for (most) of the patients. We also have a few patients breaching the 200s and making their way to the 400s or 500s. Could there be any underlying factor?

# In[ ]:


df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


# # Train files

# With DICOM files, we can retrieve the information as a numpy array with the DICOM attribute `ds.pixel_array` from the file which is relatively simple to use. We can invert the colors by adding a `-` before it.

# In[ ]:


r = dcm.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/11.dcm')
img = r.pixel_array
img[img == -2000] = 0

plt.axis('off')
plt.imshow(img)
plt.show()

plt.axis('off')
plt.imshow(-img) # Invert colors with -
plt.show()


# What we can do is **visualize** the DICOM files with a simple glob.glob for the train and test set. It is remarkable that we can extract this information from a DICOM.

# In[ ]:


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)
            dataset.PixelSpacing = [1, 1]
        plt.figure(figsize=(10, 10))
        plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
        plt.show()
for file_path in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm'):
    dataset = dcm.dcmread(file_path)
    show_dcm_info(dataset)
    break # Comment this out to see all


# This lung image looks very similar to the Data Science Bowl 2017 and it also looks like we have quite a bit of metadata coming with each DICOM file. Now let's examine a couple of images at a glance:

# In[ ]:


files = glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm')
def dicom_to_image(filename):
    im = dcm.dcmread(filename)
    img = im.pixel_array
    img[img == -2000] = 0
    return img
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
for i in range(20):
    plots[i // 5, i % 5].axis('off')
    plots[i // 5, i % 5].imshow(dicom_to_image(files[i]), cmap=plt.cm.bone)


# All the images look fairly uniform - how about taking into account randomness?

# In[ ]:


files = glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm')
def dicom_to_image(filename):
    im = dcm.dcmread(filename)
    img = im.pixel_array
    img[img == -2000] = 0
    return img
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
for i in range(20):
    plots[i // 5, i % 5].axis('off')
    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files)), cmap=plt.cm.bone)


# So some images are uniform but others are not.

# # Train CSV exploration

# Let's look at the CSV metadata now.

# In[ ]:


import plotly.io as pio
pio.templates.default = "ggplot2"
xcoord = df_train["Age"].value_counts().keys()
y1 = df_train["Age"].value_counts().values


annotationsList = [dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor='right',
                yanchor='bottom',
                showarrow=False,
            ) for xi, yi in zip(xcoord, y1)]


annotations = annotationsList
data=[go.Bar(
    x=list(df_train['Age'].value_counts().keys()), 
    y=list(df_train['Age'].value_counts().values)

)]
layout=go.Layout(height=800, width=800, title='Distribution of training labels', annotations=annotations)
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='train-label-dist')


# How about the duration? HMMMMM....

# In[ ]:


import plotly.express as px
data=px.bar(x=list(df_train['Weeks'].value_counts().keys()), y=list(df_train['Weeks'].value_counts().values))
data


# So it seems like we have a rather lackadaisical distribution of Age in the data which is interesting to consider. How about the target distribution in the data? (FVC)

# In[ ]:


plt.style.use('seaborn-darkgrid')
data=plt.scatter(x=list(df_train['FVC'].value_counts().keys()), y=list(df_train['FVC'].value_counts().values))
data;


# So as of now, let's take a look at smoking status of each affected person (i.e whether the person is a former smoker, whether the person never smoked, or whether the person currently smokes) for perspective.

# In[ ]:


cnts = df_train['SmokingStatus'].value_counts()
cnts = cnts/cnts.sum()    # convert to percentage


# Plot
# Set order and colors
sns.set()
pref_order = ['Ex-smoker', 'Never smoked', 'Currently smokes']
pref_color = ['#F7819F', '#F5A9BC', '#E6E6E6']

# matplotlib general settings
fig, ax = plt.subplots(figsize=(20,1))
plt.title('Smoking Status', fontsize=18, loc='left')
ax.get_xaxis().set_visible(False)
ax.tick_params(axis='y', labelsize=16, labelcolor='grey')  
ax.set_facecolor('white')

# Draw each bar and text separately with appropriate offset
bar_start = 0
for i in pref_order:
    ax.barh(y=[3], width=cnts[i], height=0.1, left=bar_start, color=pref_color[pref_order.index(i)])
    #plt.text(bar_start + (cnts[i])/2 - 0.015, 0.4, "{:.0%}".format(cnts[i]), fontsize=16, transform=ax.transAxes)
    bar_start += cnts[i]

# Draw legend and set color of its text
leg = ax.legend(pref_order, loc=(0.18,-0.5), ncol=5, fontsize=14, frameon=True, facecolor='white');
for txt in leg.get_texts():
    plt.setp(txt, color='grey')


# The majority of people here are ex-smokers (approximately a third of people) and as such, we can explore further based on gender (i.e. percentage of each per male and female).

# In[ ]:


categories = ['Male', 'Female']

# Empty df to be built out
cnts = pd.DataFrame(columns = categories)

# Loop over all age categories and get distribution of responses 
for cat in categories:
    cnts[cat] = df_train.loc[df_train['Sex'] == cat, 'SmokingStatus'].value_counts()

# Drop those with no opinion
cnts = cnts/cnts.sum()    # convert to percentage


# Plot

# matplotlib settings
fig, ax = plt.subplots(figsize=(20,3))
plt.title('Smoking status per gender', fontsize=18, loc='left')
ax.get_xaxis().set_visible(False)
ax.tick_params(axis='y', labelsize=16, labelcolor='grey')  
ax.set_facecolor('white')

# Draw each bar and text separately with appropriate offset
for cat in categories:
    bar_start = 0
    for i in pref_order:
        ax.barh(y=[cat], width=cnts.loc[i,cat], height=0.6, left=bar_start, color=pref_color[pref_order.index(i)])
        bar_start += cnts.loc[i,cat]

# Draw legend and set color of its text
leg = ax.legend(pref_order, loc=(0.18,-0.2), ncol=5, fontsize=14, frameon=True, facecolor='white');
for txt in leg.get_texts():
    plt.setp(txt, color='grey')


# There's a huge difference between male and female over here in that the majority of males affected are former smokers and the majority of females affected have never smoked.

# # Fast.ai! (yay :3)

# Now with Fast.ai, it becomes a lot easier for us to try and handle the data. We can work with something called "systematic windowing" which was previously used by radiologists with regards to the brain (see the excellent notebook by Jemery Howard here: https://www.kaggle.com/jhoward/don-t-see-like-a-radiologist-fastai ) and we can easily view the details of the lungs with systematic windowing.

# In[ ]:


fn = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00026637202179561894768')
fname = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/13.dcm')
dcom = fname.dcmread()
dcom.show(scale=dicom_windows.lungs)


# Well, windowing works particularly well for computers (because us humans require a lot more than simple black, white and grey) so we can try some sort of "better" windowing in a rainbow colormap. The background ought to be black instead of a weird psychedelic *Neon Genesis Evangelion* style but it works well. A warning is due: it is shockingly bright.

# In[ ]:


dcom.show(cmap='turbo', figsize=(6,6))


# The turbo colormap which I have copy-pasted at the top of this notebook is shockingly bright, but it exposes the amount of empty space that we have in the image.

# In[ ]:


mask = dcom.mask_from_blur(dicom_windows.lungs)
wind = dcom.windowed(*dicom_windows.lungs)

_,ax = subplots(1,1)
show_image(wind, ax=ax[0])
show_image(mask, alpha=0.5, cmap=plt.cm.Reds, ax=ax[0]);


# Here we can see the distinctions.

# In[ ]:


bbs = mask2bbox(mask)
lo,hi = bbs
show_image(wind[lo[0]:hi[0],lo[1]:hi[1]]);


# Now we can isolate the lungs from the empty space.

# Now that you have got over your eyes going kaboom, let us proceed to a more, shall I say, medical, view of things with the courtesy of fast.ai and Jeremy Howard.

# In[ ]:


path = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


# The above basically just fixes up the pixel representation attribute of the DICOM file for us to use.

# In[ ]:


def dcm_tfm(fn): 
    fn = (path/fn).with_suffix('.dcm')
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        pass
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    px = x.scaled_px
    return TensorImage(px.to_3chan(dicom_windows.lungs,dicom_windows.subdural, bins=None))


# This basically reads in a DICOM file, fixes up its pixel representation attribute (as I have displayed earlier with the previous one) and it then returns a Tensor image. It fixes up our files for a neat data loader in case we ever want to use it in our modeling pipelines. This also belongs to Jeremy Howard: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai ).

# In[ ]:


show_images(dcm_tfm('1'))


# The above is basically just our earlier image in Tensor format for a neural network to utilize.

# In[ ]:


px = dcom.windowed(*dicom_windows.lungs)
show_image(px);


# This crops out the above neatly to leave just the lungs remaining right where we want them. However, we must be careful with loss of information as not even a robot would be willing to waste its valuable time examining miniscule patches on digital DICOM images.

# In[ ]:


_,axs = subplots(1,2)
dcom.show(ax=axs[0]);   dcom.show(dicom_windows.lungs, ax=axs[1])


# This function however, has a rather unique purpose which is to display our DICOM files but with a twist: it first displays the normal image and then it displays our windowed image (scroll up to check what is windowing or go to the RSNA comp and look at a couple of notebooks). We can try a gaussian blur to smooth out the bad parts here.

# In[ ]:


gdcm = gauss_blur2d(dcom.windowed(*dicom_windows.brain), 100) # using the brain for visualization purposes
show_image(gdcm);


# ..and now to shamelessly use an idea from Jeremy (again) and select the areas bright enough in the picture (segment it)

# In[ ]:


show_image(gdcm>0.3);


# Now create a simple mask for our image which (albeit looks weird without the main image) will serve certain purposes for us when it comes to modeling.

# In[ ]:


show_image(dcom.mask_from_blur(dicom_windows.lungs), cmap=plt.cm.Reds, alpha=0.6);


# All this does is reduce the image to a smaller one *without* the unnecessary area at the bottom of the image.

# In[ ]:


def pad_square(x):
    r,c = x.shape
    d = (c-r)/2
    pl,pr,pt,pb = 0,0,0,0
    if d>0: pt,pd = int(math.floor( d)),int(math.ceil( d))        
    else:   pl,pr = int(math.floor(-d)),int(math.ceil(-d))
    return np.pad(x, ((pt,pb),(pl,pr)), 'minimum')

def crop_mask(x):
    mask = x.mask_from_blur(dicom_windows.lungs)
    bb = mask2bbox(mask)
    if bb is None: return
    lo,hi = bb
    cropped = x.pixel_array[lo[0]:hi[0],lo[1]:hi[1]]
    x.pixel_array = pad_square(cropped)
_,axs = subplots(1,2)
dcom.show(ax=axs[0])
crop_mask(dcom)
dcom.show(ax=axs[1]);


# As we have seen, it is possible to toy with DICOM files. However, now we can convert a DICOM to a data frmae.

# In[ ]:


df = pd.DataFrame.from_dicoms(fn.ls())
df.head()


# And toy with the data (again)

# In[ ]:


import plotly.express as px
selected_data = df
N = len(df.img_mean)
trace1 = go.Scatter3d(
    x=selected_data.img_mean.values[0:N], 
    y=selected_data.img_std.values[0:N],
    z=selected_data.img_pct_window.values[0:N],
    mode='markers',
    marker=dict(
        colorscale = "Jet",
        colorbar=dict(thickness=10, title="image columns", len=0.8),
        opacity=0.4,
        size=2
    )
)

figure_data = [trace1]
layout = go.Layout(
    title = "Visualizing Mean, Stddev and Window values",
    scene = dict(
        xaxis = dict(title="Image mean"),
        yaxis = dict(title="Image standard deviation"),
        zaxis = dict(title="Image window values"),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# Now that it is possible for us to visualize the distribution, it's actually possible for us to view it as a correlation matrix with Plot.ly of course (I am bored of Seaborn :p).

# Now we can view the distributions of image statistics one-by-one:

# In[ ]:


def distrib_summ(t):
    plt.hist(t,40)
    return array([t.min(),*np.percentile(t,[0.1,1,5,50,95,99,99.9]),t.max()], dtype=np.int)
distrib_summ(df.img_max.values)


# Not such a comfortable distribution to be working with....

# In[ ]:


distrib_summ(df.img_std.values)


# Rather odd distribution over here too.

# In[ ]:


distrib_summ(df.img_pct_window.values)


# Now to move on to preprocessing.

# # Preprocessing (credits to Guido Zuidhof)

# The functions and 3d plotting are from Guido's notebook: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial and I've tried to add clarity wherevr possible to whatever Guido has done here.

# In[ ]:


import os
import pydicom as dicom, numpy as np
from matplotlib import pyplot as plt
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# So what this does is basically get the dicom files, sort the files and get slice thickness and make it a file property (or modify the SliceThickness attribute if the attribute exists).

# In[ ]:


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# This gets the Hounsfield Units (a measurement to which the CT scanners are delicately calibrated to) and Guido calls it:
# > a measure of radiodensity
# 
# So for those of you unenlightened on the subject of radiodensity, I looked it up quickly, and to quote Wikipedia:
# > Radiodensity (or radiopacity) is opacity to the radio wave and X-ray portion of the electromagnetic spectrum: that is, the relative inability of those kinds of electromagnetic radiation to pass through a particular material.
# 
# In layman's terms, it's the ability of a certain kind of material to prevent radio waves or X-rays to pass through them.

# In[ ]:


first_patient = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
first_patient_pixels = get_pixels_hu(first_patient)
fig, ax = plt.subplots()
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")

plt.show()


# We also can visualize convolution filters:

# In[ ]:


import cv2
img = dicom.dcmread('../input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/1.dcm')
def conv(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((7, 7), np.float32)/25
    conv = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title('Convolved Image', fontsize=24)
    plt.show()
conv(img.pixel_array)


# # Afterword: a better preprocessing?
# 
# So yes, we do need to preprocess a load of images, considering the discrepancies between most of our train images, especially with regards to the border (which is either circular or nonexistent in most cases). We might need to even this out, so in that case I'll be taking the aid of Roman's microscope augmentation (again another idea from SIIM-ISIC competition) over here: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159476 .
# 
# As you can see in the following image from the notebook [Your Starter Notebook for OSIC!](https://www.kaggle.com/twinkle0705/your-starter-notebook-for-osic) we have the circular/nonexistent border discrepancy. So again, we can treat this with Roman's microscope augmentation, as in the SIIM-ISIC competition the same thing existed.

# In[ ]:


import random
class Microscope:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),
                        (img.shape[0]//2, img.shape[1]//2),
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                        (0, 0, 0),
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)

        return img
    
import pydicom
dcom_file = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm')
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].set_title('Before the Aug')
ax[0].imshow(dcom_file.pixel_array, cmap='turbo')
ax[1].set_title('After the Aug')
micro = Microscope(p=0.5)
ax[1].imshow(micro(dcom_file.pixel_array), cmap='turbo')
ax[0].axis('off')
ax[1].axis('off')


# Now, however, we have normalized the images and found a pretty good colormap to use. So that takes care of our first problem. The second problem however is to see how we can improve on Roman's work. Now we can turn to the methods of Neuron Engineer, whose preprocessing methods have been enumerated on many times previously by other notebooks, including mine on SIIM-ISIC. 
# 
# He's using Ben Graham's technique (which won a first place in the diabetic retinopathy competition) and if you want to look deeper in the preprocessing methods, refer to the following link: https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
# 
# Now here we can use both Ben Graham's and Roman's techniques in conjunction with each other, by performing the microscope augmentation technique and then adding a gaussian blur from Ben's work. Normally, adding a Gaussian blur would be much more difficult - if not for fast.ai and the magic of its medical imaging module.

# In[ ]:


import tempfile
import datetime

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset

# Create some temporary filenames
suffix = '.dcm'
filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name
filename_big_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

print("Setting file meta information...")
# Populate required values for file meta information
file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
file_meta.MediaStorageSOPInstanceUID = "1.2.3"
file_meta.ImplementationClassUID = "1.2.3.4"

print("Setting dataset values...")
# Create the FileDataset instance (initially no data elements, but file_meta
# supplied)
ds = FileDataset(filename_little_endian, {},
                 file_meta=file_meta, preamble=b"\0" * 128)

# Add the data elements -- not trying to set all required here. Check DICOM
# standard
ds.PatientName = "Test^Firstname"
ds.PatientID = "123456"

# Set the transfer syntax
ds.is_little_endian = True
ds.is_implicit_VR = True

# Set creation date/time
dt = datetime.datetime.now()
ds.ContentDate = dt.strftime('%Y%m%d')
timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
ds.ContentTime = timeStr

print("Writing test file", filename_little_endian)
ds.save_as(filename_little_endian)
print("File saved.")
ds.PhotometricInterpretation = 'MONOCHROME2'
ds.PixelRepresentation = 0
ds.SamplesPerPixel = 1
# Write as a different transfer syntax XXX shouldn't need this but pydicom
# 0.9.5 bug not recognizing transfer syntax
ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
ds.is_little_endian = False
ds.is_implicit_VR = False
ds.pixel_array = micro(dcom_file.pixel_array)
ds.BitsAllocated = len(micro(dcom_file.pixel_array))
print("Writing test file as Big Endian Explicit VR", filename_big_endian)
ds.save_as(filename_big_endian)
fn = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/')
fname = Path(filename_big_endian)
dcom = fname.dcmread()
dcom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
dcom.pixel_array = micro(dcom_file.pixel_array)
dcom.BitsAllocated = 16
dcom.PhotometricInterpretation = 'MONOCHROME2'
dcom.PixelRepresentation = 0
dcom.SamplesPerPixel = 1
dcom.Modality = 'CT'
dcom.RescaleSlope = "1.0"
dcom.RescaleIntercept = "-1024.0"
gdcm = gauss_blur2d(dcom.windowed(*dicom_windows.brain), 14) # using the brain for visualization purposes
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ax[0].set_title('Before the Aug')
ax[0].imshow(dcom_file.pixel_array, cmap='turbo')
ax[1].set_title('After the Aug')
micro = Microscope(p=0.5)
ax[1].imshow(gdcm, cmap='turbo')
ax[0].axis('off')
ax[1].axis('off')


# So this is the complete merger of both Roman's and Ben's augmentations, you can use this "slightly" blurred image for your modeling purposes, and since it's been microscoped it would work pretty nicely.

# In[ ]:


import gc;gc.collect()


# # Building a baseline model
# 
# 

# Here are some objectives of our *baseline* model that we will add to it:
# + **Some basic transforms**: We can use some simple `torchvision` transforms w/ our model, such as ToTensor and torchvision normalization. This will help our model to potentially generalize well, what with the diversity we're adding to our trianing data and what not.
# + **Simple CNN-based model**: This will be using a simple convolutional neural network-based model w/ the assistance of PyTorch, and I will also attempt clumsily to demonstrate each layer in detail using the torchsummary package.

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# I initially wanted to use [Carlos Souza's excellent End-to-end model notebook's data generator](https://www.kaggle.com/carlossouza/end-to-end-model-ct-scans-tabular) for the data preprocessing and generation part. However, I decided to use his preprocessed CT Scans in conjunction with his data loader for demo purposes.

# In[ ]:


class CTTensorsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.tensor_files = sorted([f for f in self.root_dir.glob('*.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = torch.load(self.tensor_files[item])
        if self.transform:
            image = self.transform(image)

        return {
            'patient_id': self.tensor_files[item].stem,
            'image': image
        }

    def mean(self):
        cum = 0
        for i in range(len(self)):
            sample = self[i]['image']
            cum += torch.mean(sample).item()

        return cum / len(self)

    def random_split(self, val_size: float):
        num_val = int(val_size * len(self))
        num_train = len(self) - num_val
        return random_split(self, [num_train, num_val])
    
train = CTTensorsDataset(
    root_dir=Path('../input/osic-cached-dataset')
)
cum = 0
for i in range(len(train)):
    sample = train[i]['image']
    cum += torch.mean(sample).item()


# Now here are the transforms: random sized crop, random flipping, convert to tensor and normalization.

# In[ ]:


dataloader = torch.utils.data.DataLoader(train, batch_size=4,
                        shuffle=True, num_workers=4)


# Now we're going to define a Model class so that we're able to train this CNN based model on our input images.

# In[ ]:


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dsconv1 = nn.Conv3d(1, 16, 3)
        self.dsconv2 = nn.Conv3d(16, 32, 3)
        self.d = nn.Dropout(p=0.2)
        self.m = nn.MaxPool3d((3, 3, 2), stride=(2, 2, 1))
        
    def forward(self, inputs):
        x = self.dsconv1(inputs)
        x = self.dsconv2(x)
        x = self.m(x)
        x = self.d(x)
        x = torch.flatten(x, 1)
        return x


# #### Explanation of model architecture
# 
# First of all, I already said that depthwise-pointwise convolutions will be the main event in our model, so as such, allow me to present to you the formula for a simple convolution:
# 
# $$out(N_i, C_{out_j}) = bias(C_{out_j}) +
#                         \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)$$
# 
# Now a depthwise convolution takes the `groups` parameter (and it declares the output shape as the input shape multiplied by kernels per layer) and defines it as the same as the input shape. The pointwise convolution does away wit the groups and then processes the depthwise convolution to return an output. 
# 
# Depthwise convolutions would be better suited for TensorFlow rather than PyTorch because of the way they are optimized in PyTorch rather than TF, however I'm solely writing this in PyTorch for learning purposes.

# # Todo:
# 
# + start training
# + more detailed EDA on image features

# Again, cheerio for now, I'll be back with more fastai for you later.
# 
# ## Thank you to Jeremy Howard.
