#!/usr/bin/env python
# coding: utf-8

# ## TensorFlow Object Detection API + Apache Spark
# 
# The goal of this notebook is to utilize TensorFlow Object Detection API [1] and provide insight on how to prepare the training files using Apache Spark [2].
# 
# TensorFlow Object Detection API is a research library maintained by Google that contains multiple pretrained, ready for transfer learning object detectors that provide different speed vs accuracy tradeoffs [3]. Examples include Faster R-CNN, YOLO and SSD.

# In[ ]:


from IPython.display import Image
Image("../input/object-detection-kite/kites_detections_output.jpg")


# In the kite example image above we can see multiple objects detected at multiple scales. Object detectors enable us to capture rich information present in the image (when a single label is not informative enough) and is the key technology behind many applications such as visual search.

# ## Installation Notes
# 1. git clone https://github.com/tensorflow/models.git
# 2. have a look at installation instructions: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md  
# Install / make sure you have 3.0.0 version of protobuf:  https://github.com/tensorflow/models/issues/4062  
# Updated your ~/.bashrc to locate the protobuf binary and the required directories  
# export PATH=PATH:<full-path-to-downloaded-protobuf-3.0.0>/bin/  
# export PYTHONPATH=$PYTHONPATH:<full path to tensorflow>/models/research/:<full path to tensorflow>/models/research/slim/    
# protoc object_detection/protos/*.proto --python_out=.   
# 
# 3. test your installation (from /tensorflow/models/research directory) should work without a problem:   
# python object_detection/builders/model_builder_test.py

# ## Preparing Data
# We'll use the PASCAL VOC data format for storing images and annotations. For training, we need to generate a TFRecord file using the API script: create\_pascal\_tf\_record.py . We'll also need to define the label map, e.g. similar to one in object\_detection/data/pascal\_label\_map.pbtxt  
# 
# For more info on how to generate TFRecord train and validation files, have a look at: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md

# ## Selecting the Model
# The model can be selected by modifying the corresponding config file located at: 
# https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
# 
# In particular, for transfer learning, we'd like to modify the number of classes and all the paths marked to be configured. We also need to download the check-point files from the model zoo, for a list of available pre-trained models and to download see:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md  
# 

# ## Training and Evaluation 
# To train and evaluate the model, we need to copy the following files (created above) into a training folder:  
# *dataset\_train.record*, *dataset\_val.record*, *model\_name.config*, *label\_map.pbtxt*, *model.ckpt* in addition to *train.py* and *eval.py*  
#     
# Once we have all the files, we are ready to train and evaluate the model as follows:
# > python object_detection/train.py --logtostderr  --pipeline\_config\_path={PATH\_TO\_CONFIG}  
# --train\_dir=${PATH\_TO\_TRAIN\_DIR}  
#  
# > python object_detection/eval.py --logtostderr --pipeline_config_path={PATH\_TO\_CONFIG} --checkpoint_dir={PATH\_TO\_TRAIN\_DIR} --eval_dir={PATH\_TO\_EVAL\_DIR}
#  
# > tensorboard --logdir=${PATH\_TO\_MODEL\_DIR}
#  
# > python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path {PATH\_TO\_CONFIG}   
# --trained_checkpoint_prefix {PATH\_TO\_TRAIN\_DIR}/model.ckpt-XXXX --output_directory {PATH\_TO\_TRAIN\_DIR}/inference\_graph
# 
# Note that we can start training and evaluation simultaneously (in two different terminals or on two different machines / GPUs).  
# After the model has been trained, we need to export it using TensorFlow graph proto as described here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md
# 
# 

# ## Multi-GPU training
# 
# TensorFlow Object Detection API supports multi-GPU training that can be enabled as follows (where num_clones  = number of GPUs):
# > python object\_detection/train.py --logtostderr --pipeline\_config\_path={PATH\_TO\_CONFIG} --train\_dir={PATH\_TO\_TRAIN\_DIR}  
#     --num\_clones=4 --ps\_tasks=1

# ## Notes on Evaluation
# The metric used to evaluate object detection performance is **mean average precision** (mAP@IoU). mAP is computed by taking the average of AP over the object detection classes. Average Precision (AP) is computed per class by fixing the IoU threshold (e.g. 0.5) and calculating precision (fraction of bounding boxes that overlap with ground truth by at least the IoU threshold). To learn more about evaluation metrics for object detection, have a look at [4].  
# 
# In order to configure the IoU threshold for evaluation, you have to modify matching_iou_threshold parameter in object\_detection/utils/object\_detection\_evaluation.py
# 
# In addition, in order to display detection results (especially during the beggining of training), we might want to lower the score threshold:
# 
# > eval_config: {  
#   num_examples: 20000  
#   num_visualizations: 16  
#   min_score_threshold: 0.15  
#   max_evals: 1  
# }  
# 
# By default the evaluator is set to EVAL\_DEFAULT\_METRIC = 'pascal\_voc\_detection\_metrics' in order to change it to use Open Image evaluation metrics you have to modify /models/research/object\_detection/evaluator.py script and set EVAL\_DEFAULT\_METRIC = 'open\_images\_V2\_detection\_metrics'.  
# 
# 

# ## Preparing Training Files

# In order to train the object detector, we need to prepare the training files using the following structure:
# > VOC2012  
#     |\_\_\_Annotations  
#     |\_\_\_ImageSets/Main   
#     |\_\_\_JPEGImages 
# 
# For every image file we need to create an XML annotations files with information about different objects, their labels and bounding box coordinates. In addition, ImageSets/Main folder stores the list of training, validation and trainval files that contain indicators for when a particular object is present in an image. Finally, the actual images are stored in JPEGImages folder.
#     
# 
# 

# ## Apache Spark: Preprocessing Data
# 
# Due to large dataset size, it's good to parallelize computations using Apache Spark [2]. First, we'll need to pre-process the data and extract image dimensions (since the bounding box coordinates in the dataset are normalized between 0 and 1). We can do that by defining a User Defined Function (UDF) in pySpark:
# 
#     from skimage.io import imread
#     def get_dims(filename):
#        img = imread("/path/to/VOC2012/JPEGImages/" + filename + ".jpg")
#        return img.shape[:2]
#     getDimsUDF = udf(get_dims, ArrayType(IntegerType()))`
# 
# Finally, we can transform our dataframe to have absolute bounding box pixel corrdinates as follows:
# 
#     dataframe_with_dims_unfiltered = (dataframe_with_dims
#     .withColumn("x1", (col("XMin") * col("width")).cast(IntegerType()))
#     .withColumn("y1", (col("YMin") * col("height")).cast(IntegerType()))
#     .withColumn("x2", (col("XMax") * col("width")).cast(IntegerType()))
#     .withColumn("y2", (col("YMax") * col("height")).cast(IntegerType())).cache()
#     )
#     dataframe_with_dims_unfiltered.createOrReplaceTempView("unfiltered_annotations")
# 
# Having created a temporary view, we can now use SQL to pre-process the data:
# 
#     widthThreshold = 0.001 
#     heightThreshold = 0.001
#     widthPadding = 1 
#     heightPadding = 1
#     maxSize = 6000
#     
#     spark.sql("""
#             SELECT * FROM unfiltered_annotations 
#              WHERE width < {maxSize} AND height < {maxSize} 
#              AND (x2 - x1 > {widthThreshold} * width AND y2 - y1 > {heightThreshold} * height)
#              AND (x1 < x2 AND x1 > {widthPadding} AND x2 < width - {widthPadding})
#              AND (y1 < y2 AND y1 > {heightPadding} AND y2 < height - {heightPadding})
#             """.format(maxSize=maxSize, widthThreshold=widthThreshold, heightThreshold=heightThreshold, widthPadding=widthPadding, heightPadding=heightPadding)).createOrReplaceTempView("filtered_annotations")
# 
# 
# 

# ## Apache Spark: Generating Annotations
# 
# We can use Spark transformations and actions to generate annotations. We'll use the older resilient distributed dataset (RDD) API to extract features from our filtered dataframe. To generate the XML annotations we can use the xmltodict package. We'll first extract the bounding box coordinates and labels for each object and then merge multiple objects into a single XML file. We'll distribute the computation among multiple workers and write out the XML annotations file using a foreach action as shown below: 
# 
#     def obj2dict(row):
#         return  (row['ImageID'], (row['width'], row['height'],
#         OrderedDict([('name', row['image_group']),
#                           ('pose', 'Unspecified'),
#                           ('truncated', '0'),
#                           ('difficult', '0'),
#                           ('bndbox',
#                           OrderedDict([('xmin', row['x1']),
#                                        ('ymin', row['y1']),
#                                        ('xmax', row['x2']),
#                                        ('ymax', row['y2'])]))])))
# 
#     def img2dict(image_id, img_info):
#         width, height, image_objects = img_info[0]
#         return  (image_id, OrderedDict([('annotation',
#                  OrderedDict([('folder', 'VOC2012'),
#                          ('filename', image_id + '.jpg'),
#                          ('source',
#                          OrderedDict([('database', 'open_image'),
#                                      ('annotation', 'open_image'),
#                                      ('image', 'open_image')])),
#                                      ('size',
#                                       OrderedDict([('width',  width),
#                                                    ('height', height),
#                                                    ('depth', '3')])),
#                                      ('segmented', '0'),
#                                      ('object', image_objects)]))]))
# 
#     def dict2xml(item):
#         return xmltodict.unparse(item, pretty=True)
#   
#     def saveXML(imgId, data):
#       if not os.path.exists("/path/to/VOC2012/Annotations/" + imgId + ".xml"):
#         with open("/path/to/VOC2012/Annotations/" + imgId + ".xml", "w") as f:
#           f.write(data) . 
# 
#     xml_df = dataframe_merged.rdd.map(lambda row: obj2dict(row)).groupByKey().mapValues(list)\
#                                        .map(lambda img_info: img2dict(img_info[0], img_info[1]))\
#                                        .map(lambda item: (item[0], dict2xml(item[1])))\
#                                        .map(lambda rec: (rec[0], "\n".join(rec[1].split('\n')[1:])))\
#                                        .toDF(["ImageID", "data"])\
#                                        .cache()
#                                        
#     xml_df.rdd.foreach(lambda r: saveXML(r["ImageID"], r["data"]))
# 

# ## Apache Spark: Generating ImageSet File
# 
# Finally, we need to generate the object indicator files. We can do that by diving the dataset into training and validation:
#  
#     df_train, df_val = kaggle_annotations_merged.randomSplit([0.8, 0.2], seed=0)
#     
# And add a (+1/-1) indicator for (presence / absence) of a particular category:
# 
#     joined = kaggle_annotations_merged.select(col("ImageID"), col("Label").alias("friendly"))
#     grouped = joined.groupBy("ImageID").agg(collect_set(col("friendly")).alias("Classes"))
#     categories = list(set(map(lambda r: r[0], joined.select(col("friendly")).distinct().collect())))
# 
#     def forCategory(category):
#       outputLocation = "/path/to/VOC2012/ImageSets/Main/" + category + "_trainval.txt"
# 
#       print(category +  " -> "  + outputLocation)
#       result = (joined
#       .withColumn("Score", when(col("friendly") == category, lit(+1))
#       .otherwise(lit(-1))).select("ImageID", "Score")
#       )
#  
#       with open("/path/to/VOC2012/ImageSets/Main/" + category + "_trainval.txt", "w") as f:
#         data = result.rdd.map(lambda r: (r[0], r[1])).collect()
#         for item in data:
#           f.write(str(item[0]) + " " + str(item[1]) + "\n")
#         #end for
#       #end with
#       return result
#     
#     for cat in categories:
#        forCategory(cat)
#     
# In a similar way, we can generate train and val indicator files.
# 

# ## Conclusion
# 
# In conclusion, TensorFlow Object Detection API provides a really nice framework for evaluating multiple object detectors. Together with Spark and multi-GPU training, it's possible to scale to massive datasets and achieve high mean average precision.

# ## References
# [1] TensorFlow Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection  
# [2] Databricks Apache Spark: https://databricks.com/  
# [3] An article comparing different object detectors: https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359  
# [4] An article discussing object detector evaluation metrics: https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3   
# [5] Open Images Dataset: https://storage.googleapis.com/openimages/web/index.html
