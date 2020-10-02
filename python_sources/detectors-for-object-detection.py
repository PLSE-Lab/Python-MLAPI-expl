#!/usr/bin/env python
# coding: utf-8

# # Detectors for Object detection 
# In this problem, We are asked to do object detection on very large datasets (Open Image V4), here we have 1.7M images and annotations and have 100k images in test set. For doing computation on such a large datasets one will need multiple GPUs. I am listing some of the models that can be used for start working on this problem. The lastest model I am providing link to is of late 2016 and I hope better models are available openly. I will be providing links to following models - 
# -  **Single Shot Multibox**
# -  **Yolo**
# -  **Faster RCNN**

# # Single Shot Multibox Detector (SSD)
# 
# - Pre-trained model can be found for 20 classes
# - 100x faster than YOLO and faster RCNN (they claim in paper)
# - Can be clubbed with any classification network architecture.
# 
# #### Links - 
# 1. Research Paper link - https://storage.googleapis.com/pub-tools-public-publication-data/pdf/44872.pdf
# 2. Caffe model link - https://github.com/weiliu89/caffe/tree/ssd
# 3. Keras model SSD - https://github.com/pierluigiferrari/ssd_keras
# 4. Data generator tutorial link - https://github.com/pierluigiferrari/data_generator_object_detection_2d/blob/master/data_generator_tutorial.ipynb
# 5. Tensorflow model link - https://github.com/balancap/SSD-Tensorflow
# 
# 
# # You only look once (YOLO)
# - Yolo works at 30 FPS and said to have ~57% MAP on coco dataset.
# 
# 1. Paper link - https://pjreddie.com/media/files/papers/yolo.pdf
# 2. Link to all the information - https://pjreddie.com/darknet/yolo/
# 
# 
# 
# # Faster -RCNN 
# It is said to be slower than YOLO and SSD, but if one has time and resources it can be tried.
# Link - https://github.com/rbgirshick/py-faster-rcnn
# 
# ## Hopefully this information helps :) 
# 
# 
# 
# 
# 

# 

# 
