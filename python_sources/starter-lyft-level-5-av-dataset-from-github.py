#!/usr/bin/env python
# coding: utf-8

# Copied Lyft source code from https://github.com/lyft/nuscenes-devkit/blob/master/notebooks/tutorial_lyft.ipynb
# Don't run this kernel. CPU is not enough for this kernel to RUN!!!
# 

# In[ ]:


get_ipython().system('pip install lyft-dataset-sdk')


# 
# 
# [](http://)# Lyft Level 5 AV dataset and nuScenes devkit tutorial
# 
# Welcome to the Level 5 AV dataset & nuScenes SDK tutorial!
# 
# 
# This notebook is based on the original nuScenes tutorial notebook (https://www.nuscenes.org/) and was adjusted for the Level 5 AV dataset.

# ## Introduction to the dataset structure
# 
# In this part of the tutorial, let us go through a top-down introduction of our database. Our dataset comprises of elemental building blocks that are the following:
# 
# 1. `scene` - 25-45 seconds snippet of a car's journey.
# 2. `sample` - An annotated snapshot of a scene at a particular timestamp.
# 3. `sample_data` - Data collected from a particular sensor.
# 4. `sample_annotation` - An annotated instance of an object within our interest.
# 5. `instance` - Enumeration of all object instance we observed.
# 6. `category` - Taxonomy of object categories (e.g. vehicle, human). 
# 7. `attribute` - Property of an instance that can change while the category remains the same.
# 8. `visibility` - (currently not used)
# 9. `sensor` - A specific sensor type.
# 10. `calibrated sensor` - Definition of a particular sensor as calibrated on a particular vehicle.
# 11. `ego_pose` - Ego vehicle poses at a particular timestamp.
# 12. `log` - Log information from which the data was extracted.
# 13. `map` - Map data that is stored as binary semantic masks from a top-down view.

# Let's get started! Make sure that you have a local copy of a dataset (for download instructions, see https://level5.lyft.com/dataset/). Then, adjust `dataroot` below to point to your local dataset path. If everything is set up correctly, you should be able to execute the following cell successfully.

# In[ ]:


# Load the SDK
get_ipython().run_line_magic('matplotlib', 'inline')
from lyft_dataset_sdk.lyftdataset import LyftDataset

# Load the dataset
# Adjust the dataroot parameter below to point to your local dataset path.
# The correct dataset path contains at least the following four folders (or similar): images, lidar, maps, v1.0.1-train
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
level5data = LyftDataset(data_path='.', json_path='/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)
# Not tested this module yet


# ### 1. `scene`

# Let's take a look at the scenes that we have in the loaded database. This example dataset only has one scene, but there are many more to come.

# In[ ]:


level5data.list_scenes()


# Let's look at a scene's metadata

# In[ ]:


my_scene = level5data.scene[0]
my_scene


# ### 2. `sample`

# We define `sample` as an ***annotated keyframe of a scene at a given timestamp***. A keyframe is a frame where the time-stamps of data from all the sensors should be very close to the time-stamp of the sample it points to.
# 
# Now, let us look at the first annotated sample in this scene.

# In[ ]:


my_sample_token = my_scene["first_sample_token"]
# my_sample_token = level5data.get("sample", my_sample_token)["next"]  # proceed to next sample

level5data.render_sample(my_sample_token)


# Let's examine its metadata

# In[ ]:


my_sample = level5data.get('sample', my_sample_token)
my_sample


# A useful method is  `list_sample()` which lists all related `sample_data` keyframes and `sample_annotation` associated with a `sample` which we will discuss in detail in the subsequent parts.

# In[ ]:


level5data.list_sample(my_sample['token'])


# Instead of looking at camera and lidar data separately, we can also project the lidar pointcloud into camera images:

# In[ ]:


level5data.render_pointcloud_in_image(sample_token = my_sample["token"],
                                      dot_size = 1,
                                      camera_channel = 'CAM_FRONT')


# ### 3. `sample_data`

# The dataset contains data that is collected from a full sensor suite. Hence, for each snapshot of a scene, we provide references to a family of data that is collected from these sensors. 
# 
# We provide a `data` key to access these:

# In[ ]:


my_sample['data']


# Notice that the keys are referring to the different sensors that form our sensor suite. Let's take a look at the metadata of a `sample_data` taken from `CAM_FRONT`.

# In[ ]:


sensor_channel = 'CAM_FRONT'  # also try this e.g. with 'LIDAR_TOP'
my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])
my_sample_data


# We can also render the `sample_data` at a particular sensor. 

# In[ ]:


level5data.render_sample_data(my_sample_data['token'])


# ### 4. `sample_annotation`

# `sample_annotation` refers to any ***bounding box defining the position of an object seen in a sample***. All location data is given with respect to the global coordinate system. Let's examine an example from our `sample` above.

# In[ ]:


my_annotation_token = my_sample['anns'][16]
my_annotation =  my_sample_data.get('sample_annotation', my_annotation_token)
my_annotation


# We can also render an annotation to have a closer look.

# In[ ]:


level5data.render_annotation(my_annotation_token)


# ### 5. `instance`

# Object instance are instances that need to be detected or tracked by an AV (e.g a particular vehicle, pedestrian). Let us examine an instance metadata

# In[ ]:


my_instance = level5data.instance[100]
my_instance


# We generally track an instance across different frames in a particular scene. However, we do not track them across different scenes. In this example, we have 16 annotated samples for this instance across a particular scene.

# In[ ]:


instance_token = my_instance['token']
level5data.render_instance(instance_token)


# An instance record takes note of its first and last annotation token. Let's render them

# In[ ]:


print("First annotated sample of this instance:")
level5data.render_annotation(my_instance['first_annotation_token'])


# In[ ]:


print("Last annotated sample of this instance")
level5data.render_annotation(my_instance['last_annotation_token'])


# ### 6. `category`

# A `category` is the object assignment of an annotation.  Let's look at the category table we have in our database. The table contains the taxonomy of different object categories and also list the subcategories (delineated by a period). 

# In[ ]:


level5data.list_categories()


# A category record contains the name and the description of that particular category.

# In[ ]:


level5data.category[2]


# ### 7. `attribute`

# An `attribute` is a property of an instance that may change throughout different parts of a scene while the category remains the same. Here we list the provided attributes and the number of annotations associated with a particular attribute.

# In[ ]:


level5data.list_attributes()


# Let's take a look at an example how an attribute may change over one scene

# In[ ]:


for my_instance in level5data.instance:
    first_token = my_instance['first_annotation_token']
    last_token = my_instance['last_annotation_token']
    nbr_samples = my_instance['nbr_annotations']
    current_token = first_token

    i = 0
    found_change = False
    while current_token != last_token:
        current_ann = level5data.get('sample_annotation', current_token)
        current_attr = level5data.get('attribute', current_ann['attribute_tokens'][0])['name']

        if i == 0:
            pass
        elif current_attr != last_attr:
            print("Changed from `{}` to `{}` at timestamp {} out of {} annotated timestamps".format(last_attr, current_attr, i, nbr_samples))
            found_change = True

        next_token = current_ann['next']
        current_token = next_token
        last_attr = current_attr
        i += 1


# ### 8. `sensor`

# The Level 5 dataset consists of data collected from our full sensor suite which consists of:
# - 1 x LIDAR, (up to three in final dataset)
# - 7 x cameras, 

# In[ ]:


level5data.sensor


# Every `sample_data` has a record on which `sensor` the data is collected from (note the "channel" key)

# In[ ]:


level5data.sample_data[10]


# ### 9. `calibrated_sensor`

# `calibrated_sensor` consists of the definition of a particular sensor (lidar/camera) as calibrated on a particular vehicle. Let us look at an example.

# In[ ]:


level5data.calibrated_sensor[0]


# Note that the `translation` and the `rotation` parameters are given with respect to the ego vehicle body frame. 

# ### 10. `ego_pose`

# `ego_pose` contains information about the location (encoded in `translation`) and the orientation (encoded in `rotation`) of the ego vehicle body frame, with respect to the global coordinate system.

# In[ ]:


level5data.ego_pose[0]


# ### 11. `log`
# 
# The `log` table contains log information from which the data was extracted. A `log` record corresponds to one journey of our ego vehicle along a predefined route. Let's check the number of logs and the metadata of a log.

# In[ ]:


print("Number of `logs` in our loaded database: {}".format(len(level5data.log)))


# In[ ]:


level5data.log[0]


# Notice that it contains a variety of information such as the date and location of the log collected. It also gives out information about the map from where the data was collected. Note that one log can contain multiple non-overlapping scenes.

# ### 12. `map`

# Map information is currently stored in a 2D rasterized image. Let's check the number of maps and metadata of a map.

# In[ ]:


print("There are {} maps masks in the loaded dataset".format(len(level5data.map)))


# In[ ]:


#level5data.map[0]


# The map can e.g. be displayed in the background of top-down views:

# In[ ]:


#sensor_channel = 'LIDAR_TOP'
#my_sample_data = level5data.get('sample_data', my_sample['data'][sensor_channel])
# The following call can be slow and requires a lot of memory
#level5data.render_sample_data(my_sample_data['token'], underlay_map = True)


# ## Dataset and Devkit Basics

# Let's get a bit technical.
# 
# The NuScenes class holds several tables. Each table is a list of records, and each record is a dictionary. For example the first record of the category table is stored at:

# In[ ]:


level5data.category[0]


# The category table is simple: it holds the fields `name` and `description`. It also has a `token` field, which is a unique record identifier. Since the record is a dictionary, the token can be accessed like so:

# In[ ]:


cat_token = level5data.category[0]['token']
cat_token


# If you know the `token` for any record in the DB you can retrieve the record by doing

# In[ ]:


level5data.get('category', cat_token)


# _As you can notice, we have recovered the same record!_

# OK, that was easy. Let's try something harder. Let's look at the `sample_annotation` table.

# In[ ]:


level5data.sample_annotation[0]


# This also has a `token` field (they all do). In addition, it has several fields of the format [a-z]*\_token, _e.g._ instance_token. These are foreign keys in database speak, meaning they point to another table. 
# Using `level5data.get()` we can grab any of these in constant time.
# 
# Note that in our dataset, we don't provide `num_lidar_pts` and set it to `-1` to indicate this.

# In[ ]:


one_instance = level5data.get('instance', level5data.sample_annotation[0]['instance_token'])
one_instance


# This points to the `instance` table. This table enumerate the object _instances_ we have encountered in each 
# scene. This way we can connect all annotations of a particular object.
# 
# If you look carefully at the tables, you will see that the sample_annotation table points to the instance table, 
# but the instance table doesn't list all annotations that point to it. 
# 
# So how can we recover all sample_annotations for a particular object instance? There are two ways:
# 
# 1. `Use level5data.field2token()`. Let's try it:

# In[ ]:


ann_tokens = level5data.field2token('sample_annotation', 'instance_token', one_instance['token'])


# This returns a list of all sample_annotation records with the `'instance_token'` == `one_instance['token']`. Let's store these in a set for now

# In[ ]:


ann_tokens_field2token = set(ann_tokens)

ann_tokens_field2token


# The `level5data.field2token()` method is generic and can be used in any similar situation.
# 
# 2. For certain situation, we provide some reverse indices in the tables themselves. This is one such example. 

# The instance record has a field `first_annotation_token` which points to the first annotation in time of this instance. 
# Recovering this record is easy.

# In[ ]:


ann_record = level5data.get('sample_annotation', one_instance['first_annotation_token'])
ann_record


# Now we can traverse all annotations of this instance using the "next" field. Let's try it. 

# In[ ]:


ann_tokens_traverse = set()
ann_tokens_traverse.add(ann_record['token'])
while not ann_record['next'] == "":
    ann_record = level5data.get('sample_annotation', ann_record['next'])
    ann_tokens_traverse.add(ann_record['token'])


# Finally, let's assert that we recovered the same ann_records as we did using level5data.field2token:

# In[ ]:


print(ann_tokens_traverse == ann_tokens_field2token)


# ## Reverse indexing and short-cuts
# 
# The dataset tables are normalized, meaning that each piece of information is only given once.
# For example, there is one `map` record for each `log` record. Looking at the schema you will notice that the `map` table has a `log_token` field, but that the `log` table does not have a corresponding `map_token` field. But there are plenty of situations where you have a `log`, and want to find the corresponding `map`! So what to do? You can always use the `level5data.field2token()` method, but that is slow and inconvenient. The devkit therefore adds reverse mappings for some common situations including this one.
# 
# Further, there are situations where one needs to go through several tables to get a certain piece of information. 
# Consider, for example, the category name of a `sample_annotation`. The `sample_annotation` table doesn't hold this information since the category is an instance level constant. Instead the `sample_annotation` table points to a record in the `instance` table. This, in turn, points to a record in the `category` table, where finally the `name` fields stores the required information.
# 
# Since it is quite common to want to know the category name of an annotation, we add a `category_name` field to the `sample_annotation` table during initialization of the NuScenes class.
# 
# In this section, we list the short-cuts and reverse indices that are added to the `NuScenes` class during initialization. These are all created in the `NuScenes.__make_reverse_index__()` method.

# ### Reverse indices
# The devkit adds two reverse indices by default.
# * A `map_token` field is added to the `log` records.
# * The `sample` records have shortcuts to all `sample_annotations` for that record as well as `sample_data` key-frames. Confer `level5data.list_sample()` method in the previous section for more details on this.

# ### Shortcuts

# The sample_annotation table has a "category_name" shortcut.

# _Using shortcut:_

# In[ ]:


catname = level5data.sample_annotation[0]['category_name']


# _Not using shortcut:_

# In[ ]:


ann_rec = level5data.sample_annotation[0]
inst_rec = level5data.get('instance', ann_rec['instance_token'])
cat_rec = level5data.get('category', inst_rec['category_token'])

print(catname == cat_rec['name'])


# The sample_data table has "channel" and "sensor_modality" shortcuts:

# In[ ]:


# Shortcut
channel = level5data.sample_data[0]['channel']

# No shortcut
sd_rec = level5data.sample_data[0]
cs_record = level5data.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
sensor_record = level5data.get('sensor', cs_record['sensor_token'])

print(channel == sensor_record['channel'])


# ## Data Visualizations
# 
# We provide list and rendering methods. These are meant both as convenience methods during development, and as tutorials for building your own visualization methods. They are implemented in the NuScenesExplorer class, with shortcuts through the NuScenes class itself.

# ### List methods
# There are three list methods available.

# 1. `list_categories()` lists all categories, counts and statistics of width/length/height in meters and aspect ratio.

# In[ ]:


level5data.list_categories()


# 2. `list_attributes()` lists all attributes and counts.

# In[ ]:


level5data.list_attributes()


# 3. `list_scenes()` lists all scenes in the loaded DB.

# In[ ]:


level5data.list_scenes()


# ### Render

# First, let's plot a lidar point cloud in an image. Lidar allows us to accurately map the surroundings in 3D.

# In[ ]:


my_sample = level5data.sample[10]
level5data.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')


# We can also plot all annotations across all sample data for that sample.

# In[ ]:


my_sample = level5data.sample[20]

# The rendering command below is commented out because it tends to crash in notebooks
# level5data.render_sample(my_sample['token'])


# Or if we only want to render a particular sensor, we can specify that.

# In[ ]:


level5data.render_sample_data(my_sample['data']['CAM_FRONT'])


# Additionally we can aggregate the point clouds from multiple sweeps to get a denser point cloud.

# In[ ]:


level5data.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5)


# We can even render a specific annotation.

# In[ ]:


level5data.render_annotation(my_sample['anns'][22])


# Finally, we can render a full scene as a video. There are two options here:
# 1. level5data.render_scene_channel() renders the video for a particular channel. (HIT ESC to exit)
# 2. level5data.render_scene() renders the video for all surround view camera channels.
# 
# NOTE: These methods use OpenCV for rendering, which doesn't always play nice with IPython Notebooks. If you experience any issues please run these lines from the command line. 

# In[ ]:


#my_scene_token = level5data.scene[0]["token"]
#level5data.render_scene_channel(my_scene_token, 'CAM_FRONT')


# There is also a method level5data.render_scene() which renders the video for all camera channels.

# In[ ]:


#level5data.render_scene(my_scene_token)


# Finally, let us visualize all scenes on the map for a particular location.

# In[ ]:


#level5data.render_egoposes_on_map(log_location='Palo Alto')


# ## Play with it!
# E.g.:
# 1. Plot 5 sequential lidar scans with underlying semantic map.

# In[ ]:


# put your code here
# hint: 
# next_sample_data = level5data.get('sample_data', my_sample_data["next"])
# gives you the next sample data entry


# 2. Show an annotation at the moment when an attribute changes.

# In[ ]:




# put your code here