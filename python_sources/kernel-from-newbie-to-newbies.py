#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image, ImageDraw


# Install ImageAI (A python library built to empower developers to build applications and systems with self-contained Computer Vision capabilities http://imageai.org)

# In[ ]:


get_ipython().system('pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.3/imageai-2.0.3-py3-none-any.whl')


# Copy pre-trained model (yolo.h5) for Image Recognition and Object Recognition tasks in ImageAI

# In[ ]:


get_ipython().system('wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5')


# Copy openimages class names

# In[ ]:


get_ipython().system('wget https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv')


# View input files

# In[ ]:


s_sub = pd.read_csv('../input/sample_submission.csv')
s_sub.head()


# In[ ]:


test_filename = os.listdir('../input/test')
test_filename[:5]


# In[ ]:


labelMap = pd.read_csv('class-descriptions-boxable.csv', header=None, names=['labelName', 'Label'])
labelMap.head()


# In[ ]:


# Show one image
def show_image_by_index(i):
    sample_image = plt.imread(f'../input/test/{test_filename[i]}')
    plt.imshow(sample_image)

def show_image_by_filename(filename):
    sample_image = plt.imread(filename)
    plt.imshow(sample_image)


# Test procedures

# In[ ]:


show_image_by_index(22)


# In[ ]:


show_image_by_filename(f'../input/test/e7c0991d9a37bdef.jpg')


# Import additional modules

# In[ ]:


from imageai.Detection import ObjectDetection


# In[ ]:


#Get the path to the working directory
execution_path = os.getcwd()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# load model\ndetector = ObjectDetection()\ndetector.setModelTypeAsYOLOv3()\ndetector.setModelPath(os.path.join(execution_path, "yolo.h5"))\ndetector.loadModel()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# test detection on one image\ndetections = detector.detectObjectsFromImage(input_image=os.path.join(\'../input/test\', \'e7c0991d9a37bdef.jpg\'),\n                                                                      #test_filename[64]), \n                                             output_image_path=os.path.join(execution_path , "result.jpg"),\n#                                            output_type = \'array\',\n                                             extract_detected_objects = False)\nfor eachObject in detections:\n    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )\n\n# show the result\nshow_image_by_filename(\'./result.jpg\')')


# In[ ]:


#view detection variable
detections


# In[ ]:


def format_prediction_string(image_id, result, labelMap, xSize, ySize):
    prediction_strings = []
    #print(xSize, ySize)
    for i in range(len(result)):
        class_name = result[i]['name'].capitalize()
        class_name = pd.DataFrame(labelMap.loc[labelMap['Label'].isin([class_name])]['labelName'])
        #print(result[i]['box_points'])
        xMin = result[i]['box_points'][0] / xSize
        xMax = result[i]['box_points'][2] / xSize
        yMin = result[i]['box_points'][1] / ySize
        yMax = result[i]['box_points'][3] / ySize
        
        if len(class_name) > 0:
            class_name = class_name.iloc[0]['labelName']
            boxes = [xMin, yMin, xMax, yMax]#result[i]['box_points']
            score = result[i]['percentage_probability']

            prediction_strings.append(
                f"{class_name} {score} " + " ".join(map(str, boxes))
            )
        
    prediction_string = " ".join(prediction_strings)

    return {
            "ImageID": image_id,
            "PredictionString": prediction_string
            }


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Test prediction on input images\nres = []\nfor i in tqdm(os.listdir(\'../input/test\')[20:25]):\n    detections = detector.detectObjectsFromImage(input_image=os.path.join(\'../input/test\', i),\n                                                 output_image_path=os.path.join(execution_path , "result.jpg"),\n                                                 #output_type = \'array\',\n                                                 extract_detected_objects = False)\n    currentImg = Image.open(os.path.join(\'../input/test\', i))\n    xSize = currentImg.size[0]\n    ySize = currentImg.size[1]\n    #print(xSize, ySize)\n    p = format_prediction_string(i, detections, labelMap, xSize, ySize)\n    res.append(p)')


# In[ ]:


res[1:2]


# In[ ]:


# Convert res variable to DataFrame
pred_df = pd.DataFrame(res)
pred_df.head()


# In[ ]:


# Get the file name without extension
pred_df['ImageID'] = pred_df['ImageID'].map(lambda x: x.split(".")[0])


# In[ ]:


pred_df.head()


# In[ ]:


# Run detection on test images
sample_submission_df = pd.read_csv('../input/sample_submission.csv')
image_ids = sample_submission_df['ImageId']
predictions = []
res = []
for image_id in tqdm(image_ids):
    detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test', image_id + '.jpg'),
                                                 output_image_path=os.path.join(execution_path , "result.jpg"),
                                                 #output_type = 'array',
                                                 extract_detected_objects = False)
    currentImg = Image.open(os.path.join('../input/test', image_id + '.jpg'))
    xSize = currentImg.size[0]
    ySize = currentImg.size[1]
    p = format_prediction_string(image_id, detections, labelMap, xSize, ySize)
    res.append(p)


# In[ ]:


# Save submission file
pred_df = pd.DataFrame(res)
pred_df['ImageID'] = pred_df['ImageID'].map(lambda x: x.split(".")[0])
pred_df.to_csv('result.csv', index=False)

