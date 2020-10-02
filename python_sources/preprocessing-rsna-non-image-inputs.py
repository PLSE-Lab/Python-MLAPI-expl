#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# basic imports
import os, random
from tqdm import tqdm
import pydicom
import pandas as pd
import numpy as np


# In[ ]:


# what do we have here?
print (os.listdir("../input"))


# In[ ]:


# global variables
TRAIN_DIR="../input/stage_1_train_images"
# list of training images
TRAIN_LIST=sorted(os.listdir(TRAIN_DIR))

TRAIN_LABELS_CSV_FILE="../input/stage_1_train_labels.csv"
# pedantic nit: we are changing 'Target' to 'label' on the way in
TRAIN_LABELS_CSV_COLUMN_NAMES=['patientId', 'x1', 'y1', 'width', 'height', 'label']

DETAILED_CLASSES_CSV_FILE="../input/stage_1_detailed_class_info.csv"
DETAILED_CLASSES_CSV_COLUMN_NAMES=['patientId' , 'class']
# dictionary to map string classes to numerical
CLASSES_DICT={'Normal': 0, 'Lung Opacity' : 1, 'No Lung Opacity / Not Normal' : 2}

TEST_DIR="../input/stage_1_test_images"
# list of test images
TEST_LIST=sorted(os.listdir(TEST_DIR))

IMAGE_SIZE=[1024,1024]

# intermediate meta-data files
TRAIN_METADATA_CSV_FILE="stage_1_train_metadata.csv"
TRAIN_METADATA_CSV_COLUMN_NAMES=['patientId', 'sex', 'age', 'viewPosition']
TEST_METADATA_CSV_FILE="stage_1_test_metadata.csv"
TEST_METADATA_CSV_COLUMN_NAMES=['patientId', 'sex', 'age', 'viewPosition']
# dictionaries to map sex and viewPosition fields to numerical
PATIENTSEX_DICT={'M': 0, 'F' : 1}
VIEWPOSITION_DICT={'AP': 0, 'PA' : 1}

# we will consolidate label, classes, and processed train meta-data into one csv file
TRAIN_PROCESSED_CSV_FILE="stage_1_train_processed.csv"
TRAIN_PROCESSED_CSV_COLUMN_NAMES=['patientId', 'label', 'class', 'sex', 'age', 'viewPosition']

# we will pre-process bounding boxes a bit and save in a separate csv file
TRAIN_BOUNDINGBOX_CSV_FILE="stage_1_train_boundingboxes.csv"
TRAIN_BOUNDINGBOX_CSV_COLUMN_NAMES=['patientId', 'boundingbox']

# we will save processed test meta-data into one csv file
TEST_PROCESSED_CSV_FILE="stage_1_test_processed.csv"
TEST_PROCESSED_CSV_COLUMN_NAMES=['patientId', 'sex', 'age', 'viewPosition']


# In[ ]:


# how many unique training examples (patient images) do we have?
print ("Unique training examples provided: {}".format(len(os.listdir(TRAIN_DIR))))


# In[ ]:


# remember 25684!


# In[ ]:


# how many test cases we have to analyze?
print ("Test cases to be predicted: {}".format(len(os.listdir(TEST_DIR))))


# In[ ]:


# read a random training patient data file
trainpatientdata = pydicom.dcmread(os.path.join(TRAIN_DIR, random.choice(TRAIN_LIST)))
# print patient dataset
print("Training Patient Dataset: \n.{}\n".format(trainpatientdata))
# print patient dataset attributes
print("Training Patient Dataset Attributes: \n{}\n".format(trainpatientdata.dir()))


# In[ ]:


# PatientSex, PatientAge, and ViewPosition look like useful attributes of trainpatientdata.
# the image is in trainpatientdata.pixel_array.


# In[ ]:


# utility to check dicom files
def checkImages (directory, filelist):
    for i, filename in tqdm(enumerate(filelist)):
        # get patientid from filename
        patientidfromfilename = filename.split(".")[0]
        # load patient meta-data only from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename), stop_before_pixels=True)
        # get patientid from patient data
        patientidfromfile=patientdata.PatientID
        # make sure everything is ok
        assert patientidfromfilename == patientidfromfile, "Error: Patient ID mismatch"


# In[ ]:


# make sure training images are ok (no assertion errors)
# need to check once only, comment out after first run
checkImages(TRAIN_DIR, TRAIN_LIST)


# In[ ]:


# make sure test images are ok (no assertion errors)
# need to check once only, comment out after first run
checkImages(TEST_DIR, TEST_LIST)


# In[ ]:


# utility to load a dicom image and/or key attributes
def loadImage (directory, filename, mode="metadata"):
    imagearray=np.zeros(IMAGE_SIZE)
    patientid= filename.split(".")[0]
    
    if mode=="metadata":
        # load patient meta-data only from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename), stop_before_pixels=True)
    elif mode=="image":
        # load patient meta-data and image from file
        patientdata = pydicom.dcmread(os.path.join(directory, filename))
        imagearray=patientdata.pixel_array
    patientid=patientdata.PatientID
    attributes="{} {} {}".format(patientdata.PatientSex,
                                 patientdata.PatientAge,
                                 patientdata.ViewPosition)
    
    return patientid, attributes, imagearray


# In[ ]:


# utility to save meta data to csv files
def saveMetaData (directory, filelist, csvfilename):
    csvrecord="patientId,sex,age,viewPosition\n"
    for filename in tqdm(filelist):
        patientid, attributes, imagearray=loadImage(directory, filename, mode="metadata")
        patientsex, patientage, patientposition = attributes.split()
        csvrecord="{}{},{},{},{}\n".format(csvrecord, patientid, patientsex, patientage, patientposition)
    # print (csvrecord)
    with open(csvfilename,'w') as file:
        file.write(csvrecord)


# In[ ]:


# save training meta data
saveMetaData(TRAIN_DIR, TRAIN_LIST, TRAIN_METADATA_CSV_FILE)


# In[ ]:


# how many unique records did we save in stage_1_train_metadata.csv?
get_ipython().system('printf "Number of unique training meta data records stored: "; grep -v "patientId,sex,age,viewPosition" stage_1_train_metadata.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_train_metadata.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_train_metadata.csv:\\n\\n";          head -10 stage_1_train_metadata.csv')


# In[ ]:


# save test meta data
saveMetaData(TEST_DIR, TEST_LIST, TEST_METADATA_CSV_FILE)


# In[ ]:


# how many unique records did we save in stage_1_test_metadata.csv?
get_ipython().system('printf "Number of unique test meta data records stored: "; grep -v "patientId,sex,age,viewPosition" stage_1_test_metadata.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_test_metadata.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_test_metadata.csv:\\n\\n";          head -10 stage_1_test_metadata.csv')


# In[ ]:


# what does the stage_1_train_labels.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_train_labels.csv: \\n\\n";          head -10 ../input/stage_1_train_labels.csv')


# In[ ]:


# what does the stage_1_detailed_class_info.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_detailed_class_info.csv:\\n\\n";          head -10 ../input/stage_1_detailed_class_info.csv')


# In[ ]:


def combineBoundingBoxes(bboxesin):
    combinedbboxlist=[]
    # convert incoming series into numpy array
    bboxesinarray=bboxesin.values
    #print ("a: {}".format(bboxesinarray))
    # compute x1+width and y1+height for all rows
    bboxesinarray[:,2]=bboxesinarray[:,0]+bboxesinarray[:,2]
    bboxesinarray[:,3]=bboxesinarray[:,1]+bboxesinarray[:,3]
    #print ("b: {}".format(bboxesinarray))
    # compute smallest and the largest x,y values across rows
    x1min, y1min, _, _ = bboxesinarray.min(axis=0)
    _, _, x2max, y2max = bboxesinarray.max(axis=0)
    #print ("c: {},{},{},{}".format(x1min, y1min, x2max, y2max))
    # insert values back in x,y,width,height format
    combinedbboxlist.insert(0, x1min)
    combinedbboxlist.insert(1, y1min)
    combinedbboxlist.insert(2, x2max-x1min)
    combinedbboxlist.insert(3, y2max-y1min)
    #print ("d: {}".format(combinedbboxlist))
    return combinedbboxlist


# In[ ]:


# read TRAIN_LABELS_CSV_FILE into a pandas dataframe
labelsbboxdf = pd.read_csv(TRAIN_LABELS_CSV_FILE,
                           names=TRAIN_LABELS_CSV_COLUMN_NAMES,
                           # skip the header line
                           header=0,
                           # index the dataframe on patientId
                           index_col='patientId')
print (labelsbboxdf.shape)
print (labelsbboxdf.head(n=10))


# In[ ]:


# grab labels by unique patienId
labelsdf=pd.DataFrame(labelsbboxdf.pop('label'), columns=['label'])
# remove duplicates
labelsdf=pd.DataFrame(labelsdf.groupby(['patientId'])['label'].first(), columns=['label'])
print (labelsdf.head(n=10))


# In[ ]:


# after 'label' is popped off, x1,y1,width,height are left in labelsbboxdf
print(labelsbboxdf.head(n=10))
bboxesdf=pd.DataFrame(labelsbboxdf.dropna())
# drop rows with NaNs (will drop rows others than the one with class 'Lung Opacity')
print(bboxesdf.head(n=10))


# In[ ]:


# compute the largest bounding box by patientId
combinedbboxdf=pd.DataFrame(bboxesdf.groupby(['patientId']).apply(combineBoundingBoxes), columns=['combinedboundingbox'])
# patch up bug in pandas resulting in the first entry being incorrect
combinedbboxdf.iat[0,0]=[264.0, 152.0, 554.0, 453.0]
combinedbboxdf.head(n=10)


# In[ ]:


# consolidate train labels, classes, and meta-data by unique patientId
# combine x,y,width,height records into one bounding box
# process test meta-data by unique patientId

# read TRAIN_LABELS_CSV_FILE into a pandas dataframe
labelsbboxdf = pd.read_csv(TRAIN_LABELS_CSV_FILE,
                           names=TRAIN_LABELS_CSV_COLUMN_NAMES,
                           # skip the header line
                           header=0,
                           # index the dataframe on patientId
                           index_col='patientId')
#print (labelsbboxdf.shape)
print (labelsbboxdf.head(n=10))



# massage bounding box information
"""x=pd.DataFrame(labelsbboxdf.pop('x'), columns=['x'])
y=pd.DataFrame(labelsbboxdf.pop('y'), columns=['y'])
width=pd.DataFrame(labelsbboxdf.pop('width'), columns=['width'])
height=pd.DataFrame(labelsbboxdf.pop('height'), columns=['height'])"""

# check the data types the import happened in (should be float64)
# print("{}\n{}\n{}\n{}".format(x.dtypes, y.dtypes, width.dtypes, height.dtypes))


# combine x, y, width, height into a consolidated bounding box dataframe
print (labelsbboxdf.head(n=10))
labelsbboxdf['x2'] = labelsbboxdf['x1']+labelsbboxdf['width']
labelsbboxdf['y2'] = labelsbboxdf['y1']+labelsbboxdf['height']
print (labelsbboxdf.head(n=10))
"""bboxdf=pd.DataFrame(x['x'].astype(np.float)+' '
                    +y['y'].astype(np.float)+' '
                    +width['width'].astype(np.float)+' '
                    +height['height'].astype(np.float), columns=['boundingbox'])
# cleanup the missing values
bboxdf=bboxdf.replace('nan nan nan nan', np.NaN)"""
#print (bboxdf.head(n=10))
# we will not remove duplicates for now and we will save bounding boxes separately
# save bounding boxes to csv file TRAIN_BOUNDINGBOX_CSV_FILE
bboxdf.to_csv(TRAIN_BOUNDINGBOX_CSV_FILE)

# read DETAILED_CLASSES_CSV_FILE into a pandas dataframe
classesdf = pd.read_csv(DETAILED_CLASSES_CSV_FILE,
                        names=DETAILED_CLASSES_CSV_COLUMN_NAMES,
                        # skip the header line
                        header=0,
                        # index the dataframe on patientId
                        index_col='patientId')
#print (classesdf.shape)
#print (classesdf.head(n=10))

# make classes numerical based on CLASSES_DICT
classesdf=classesdf.replace(to_replace=CLASSES_DICT)
# remove duplicates
classesdf=pd.DataFrame(classesdf.groupby(['patientId'])['class'].first(), columns=['class'])
#print (classesdf.head(n=10))

# read TRAIN_METADATA_CSV_FILE into a pandas dataframe
trainmetadf = pd.read_csv(TRAIN_METADATA_CSV_FILE,
                          names=TRAIN_METADATA_CSV_COLUMN_NAMES,
                          # skip the header line
                          header=0,
                          # index the dataframe on patientId
                          index_col='patientId')
#print (trainmetadf.shape)
#print (trainmetadf.head(n=10))

# make sex numerical based on PATIENTSEX_DICT
trainmetadf['sex']=trainmetadf['sex'].replace(to_replace=PATIENTSEX_DICT)
# make viewPosition numerical based on VIEWPOSITION_DICT
trainmetadf['viewPosition']=trainmetadf['viewPosition'].replace(to_replace=VIEWPOSITION_DICT)
#print (trainmetadf.head(n=10))

# consolidate label, class, and meta-data into a consolidated dataframe
trainprocesseddf=pd.concat([labelsdf, classesdf, trainmetadf], axis=1)
# save to csv file TRAIN_PROCESSED_CSV_FILE
trainprocesseddf.to_csv(TRAIN_PROCESSED_CSV_FILE)

# read TEST_METADATA_CSV_FILE into a pandas dataframe
testmetadf = pd.read_csv(TEST_METADATA_CSV_FILE,
                         names=TEST_METADATA_CSV_COLUMN_NAMES,
                         # skip the header line
                         header=0,
                         # index the dataframe on patientId
                         index_col='patientId')
#print (testmetadf.shape)
#print (testmetadf.head(n=10))
                                 
# make sex numerical based on PATIENTSEX_DICT
testmetadf['sex']=testmetadf['sex'].replace(to_replace=PATIENTSEX_DICT)
# make viewPosition numerical based on VIEWPOSITION_DICT
testmetadf['viewPosition']=testmetadf['viewPosition'].replace(to_replace=VIEWPOSITION_DICT)
#print (testmetadf.head(n=10))
# save to csv file TEST_PROCESSED_CSV_FILE
testmetadf.to_csv(TEST_PROCESSED_CSV_FILE)


# In[ ]:


# how many unique records did we save in stage_1_train_boundingboxes.csv?
get_ipython().system('printf "Number of bounding box data records (not unique) stored: "; grep -v "patientId,boundingbox" stage_1_train_boundingboxes.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_train_boundingboxes.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_train_boundingboxes.csv:\\n\\n";          head -10 stage_1_train_boundingboxes.csv')


# In[ ]:


# how many unique records did we save in stage_1_train_processed.csv?
get_ipython().system('printf "Number of unique train processed data records stored: "; grep -v "patientId,label,class,sex,age,viewPosition" stage_1_train_processed.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_train_processed.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_train_processed.csv:\\n\\n";          head -10 stage_1_train_processed.csv')


# In[ ]:


# how many unique records did we save in stage_1_test_processed.csv?
get_ipython().system('printf "Number of unique train processed data records stored: "; grep -v "patientId,sex,age,viewPosition" stage_1_test_processed.csv | sort | uniq| wc -l')


# In[ ]:


# what does the stage_1_test_processed.csv file look like?
get_ipython().system('printf "First 10 rows, including header, in stage_1_test_processed.csv:\\n\\n";          head -10 stage_1_test_processed.csv')

