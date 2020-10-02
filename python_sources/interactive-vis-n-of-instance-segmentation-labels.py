#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For dependencies
get_ipython().system('pip install -U lyft_dataset_sdk  # To load the lidar data')


# In[ ]:


# Same as all the other notebooks, symlink the directories to work with LyftDataset API
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')


# Let's get start with the data from the a random sample. According to the [official doc](https://github.com/lyft/nuscenes-devkit#dataset-structure), a `sample` is a key frame from a 25-45 seconds drive of the car. Each `sample` is accompanied with lidar data from [three different lidar sensors](https://level5.lyft.com/dataset/#data-collection), camera data from seven different cameras, and annotations of 3D objects. In this notebook, we will try to get an interactive visualization of the lidar data with [open3d](https://github.com/intel-isl/Open3D).
# 
# To be more specific, we will visualize only the top lidar in this notbook. Left and right lidars can be visualized in the same way.

# In[ ]:


from lyft_dataset_sdk.lyftdataset import LyftDataset

data_set = LyftDataset(data_path=".", json_path="data")

# We use the first 'sample' as an example ;D
sample = data_set.sample[0]

# Get the meta data for the lidar data
lidar_top = data_set.get('sample_data', sample["data"]["LIDAR_TOP"])
lidar_top


# To be able to visualize the lidar point cloud, we will first load the points from disk and prepare them as the `PointCloud` data type of `open3d`.

# In[ ]:


from pathlib import Path
import numpy as np
import open3d as o3d
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

# Load the lidar point clouds 
lidar_file = data_set.get('sample_data', sample["data"]["LIDAR_TOP"])["filename"]
lidar_pc = LidarPointCloud.from_file(Path("") / lidar_file)

# Prepare as open3d.geometry.PointCloud 
lidar_np = lidar_pc.points.transpose((1, 0))  # transpose from (3, n) to (n, 3)
lidar_xyz = lidar_np[:, :3]
lidar_intensity = lidar_np[:, 3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_xyz)
pcd


# We can see that there are 62937 points in this point cloud. Now it's time to start visualize 'em. Note that You will need to fork and run it yourself to show and interact with the plot!

# In[ ]:


from open3d import JVisualizer

visualizer = JVisualizer()
visualizer.add_geometry(pcd)
visualizer.show()


# It seems showing nothing. Don't be worried. There's a [known issue](http://www.open3d.org/docs/release/tutorial/Basic/jupyter.html#jupyter-visualization) of the fixed initial camera pose. It is not a good camera pose to visualize this point cloud. We can simply change the camera pose with the following controls:
# > - Mouse wheel: zoom in/out
# > - Left mouse button drag: rotate axis
# > - Right mouse button drag: panning
# 
# For this example, you can scroll down the mouse wheel to zoom out and the point cloud will start to show up.

# But this is NOT all of it. The visualization of Lidar points are also available from the official tool. What makes this visualization interesting is that we can customize it for special use cases, for example, visualizing instance segmentation labels!
# 
# First of all, instance segmentation labels are not provided in the training data. But given 3D Lidar points & bounding boxes, it's not difficult to extract instance segmentation labels given the fact (or assumption) that Lidar points are sparse in 3D space and bounding boxes do not suffer high overlapping with each other.
# 
# In the following code blocks, we implement a vanilla instance segmentation labels extraction function `to_ego_motion_inst_segm` and use it to extract segmentation labels per point and instance labels per point and save them into files.

# In[ ]:


import numpy as np

from pyquaternion import Quaternion
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
from lyft_dataset_sdk.utils.geometry_utils import view_points, points_in_box, quaternion_yaw


TYPE2CLASS = {
    'car':0,
    'pedestrian':1,
    'animal':2,
    'other_vehicle':3, 
    'bus':4, 
    'motorcycle':5, 
    'truck':6, 
    'emergency_vehicle':7, 
    'bicycle':8
} 

def to_ego_motion_inst_segm(sample, data_set, data_path):
    lidar_data = data_set.get('sample_data', sample["data"]["LIDAR_TOP"])
    lidar_pc = LidarPointCloud.from_file(data_path / lidar_data["filename"])

    ego_pose = data_set.get("ego_pose", lidar_data["ego_pose_token"])
    cs_pose = data_set.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    points = to_flat_vehicle_coordinates(lidar_pc, ego_pose, cs_pose)
    num_points = points.shape[1]

    lidar_xyz = points.transpose((1, 0))[:, :3]

    anns = sample["anns"]
    instance_ids = np.zeros(shape=(num_points), dtype=np.uint32)  # 0: unannotated
    label_ids = np.ones(shape=(num_points), dtype=np.uint16) * 9  # 9: unannotated
    boxes = []
    for idx, ann in enumerate(anns):
        box = data_set.get_box(ann)
        yaw = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        
        xyzlwhyawcls = np.hstack([box.center, box.wlh[[1,0,2]], (box.orientation.yaw_pitch_roll[0], TYPE2CLASS[box.name])])
        boxes.append(xyzlwhyawcls)

        # if a point belongs more than one box, assigned labels to the last one
        mask = points_in_box(box, points[:3, :])
        instance_ids[mask] = idx + 1
        label_ids[mask] = TYPE2CLASS[box.name]
    
    return lidar_xyz, np.asarray(boxes), label_ids, instance_ids

def to_flat_vehicle_coordinates(pc, pose_record, cs_record):
    """Credit goes to 
     - https://www.kaggle.com/rishabhiitbhu/eda-understanding-the-dataset-with-3d-plots/
     - https://github.com/lyft/nuscenes-devkit/
    """
    vehicle_from_sensor = np.eye(4)
    vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
    vehicle_from_sensor[:3, 3] = cs_record["translation"]
    
    ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
    rot_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
    )
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle
    points = view_points(
        pc.points[:3, :], np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor), normalize=False
    )
    return points

to_ego_motion_inst_segm


# In[ ]:


from pathlib import Path


DATA_PATH = Path(".")
output_path = DATA_PATH / "trainval"
output_path.mkdir(exist_ok=True, parents=True)
lidar_xyz, boxes, label_ids, instance_ids = to_ego_motion_inst_segm(sample, data_set, DATA_PATH)

vert_file = output_path / f"{sample['token']}_vert.npy"
np.save(vert_file, lidar_xyz)
print(f"Writing {vert_file} finished.")

bbox_file = output_path / f"{sample['token']}_bbox.npy"
np.save(bbox_file, boxes)
print(f"Writing {bbox_file} finished.")

sem_label_file = output_path / f"{sample['token']}_sem_label.npy"
np.save(sem_label_file, label_ids)
print(f"Writing {sem_label_file} finished.")

ins_label_file = output_path / f"{sample['token']}_ins_label.npy"
np.save(ins_label_file, instance_ids)
print(f"Writing {ins_label_file} finished.")


# Next, the segmentation labels/instance labels are loaded as colors for the Lidar point cloud. The colored point cloud can be visualized with open3d's JVisualizer in the same way as before.

# In[ ]:


import numpy as np
import open3d as o3d

from open3d import JVisualizer
import matplotlib.colors as mcolors

cmap = [mcolors.to_rgb(value) for value in mcolors.TABLEAU_COLORS.values()]

sample_token = sample["token"]
vert_np = np.load(output_path / f"{sample_token}_vert.npy")
seg_np = np.load(output_path / f"{sample_token}_sem_label.npy")
ins_np = np.load(output_path / f"{sample_token}_ins_label.npy")


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vert_np)
pcd.colors = o3d.utility.Vector3dVector(np.asarray([cmap[int(cls)] for cls in seg_np]))  # To visualize segmentation label
# pcd.colors = o3d.utility.Vector3dVector(np.asarray([cmap[int(ins % len(cmap))] for ins in ins_np]))  # To visualize instance label

visualizer = JVisualizer()

visualizer.add_geometry(pcd)
visualizer.show()


# That's it. Enjoy navigating the world of Lidar data!

# ## Trouble shooting
# 
# * The tested version of open3d is `open3d-python==0.7.0.0`. The newest version `open3d==0.8.0.0` is not working because of [this bug](https://github.com/intel-isl/Open3D/issues/949#issuecomment-531886936).
# * The `open3d-python` package needs to be install before the notebook start so that the jupyter widgets can be loaded. This can be done by "installing custom package" to the kernel from the settings panel.
