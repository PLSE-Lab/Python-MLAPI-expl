#!/usr/bin/env python
# coding: utf-8

# # RGB -> 3D Body Pose
# The goal is to extract body pose in 3D from videos of people running. We can then use the pose for detailed analysis of their running form

# # Pose HG 3D
# The first model I try is Pose HG 3D since it has easy to follow documentation and seems reasonably good?
# 
# Original Repo is here: https://github.com/xingyizhou/pytorch-pose-hg-3d
# 
# - **TODO**
# - Export 3D points
# - Create confidence / local region confidence data for each point

# ## Setup Environment

# In[ ]:


from IPython.display import FileLink
import os
POSE_ROOT=os.path.abspath('./pose_3d')
get_ipython().system('rm -rf {POSE_ROOT}')
os.makedirs(POSE_ROOT, exist_ok=True)
WEIGHTS_URL = 'https://www.dropbox.com/s/18uqin3dan0iknu/fusion_3d_var.pth?dl=1'
get_ipython().system('git clone https://github.com/xingyizhou/pytorch-pose-hg-3d {POSE_ROOT}')


# In[ ]:


get_ipython().system("wget -O {POSE_ROOT}/models/fusion_3d_var.pth '{WEIGHTS_URL}'")


# In[ ]:


import sys
sys.path.append(os.path.join(POSE_ROOT, 'src'))


# In[ ]:


from __future__ import absolute_import, division, print_function
import _init_paths
import os
import cv2
import numpy as np
import torch
import torch.utils.data
from opts import opts
from model import create_model
from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d


# In[ ]:


mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


# In[ ]:


from glob import glob
local_videos = glob('../input/*.mp*')


# In[ ]:


c_cap = cv2.VideoCapture(local_videos[0])
if (c_cap.isOpened()== False): 
  print("Error opening video stream or file")
ret, frame = c_cap.read()
c_cap.release()
ret, frame.shape


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.imshow(frame[:, :, ::-1])


# In[ ]:


import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
  
def show_2d(img, points, c, edges):
  num_joints = points.shape[0]
  points = ((points.reshape(num_joints, -1))).astype(np.int32)
  for j in range(num_joints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in edges:
    if points[e].min() > 0:
      cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

class Debugger(object):
  def __init__(self, ipynb=False, edges=mpii_edges):
    self.ipynb = ipynb
    self.plt = plt
    self.fig = self.plt.figure()
    self.ax = self.fig.add_subplot((111),projection='3d')
    self.ax.grid(False)
    oo = 1e10
    self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
    self.xmin, self.ymin, self.zmin = oo, oo, oo
    self.imgs = {}
    self.edges=edges
    

  
  def add_point_3d(self, points, c='b', marker='o', edges=None):
    if edges == None:
      edges = self.edges
    #show3D(self.ax, point, c, marker = marker, edges)
    points = points.reshape(-1, 3)
    x, y, z = np.zeros((3, points.shape[0]))
    for j in range(points.shape[0]):
      x[j] = points[j, 0].copy()
      y[j] = points[j, 2].copy()
      z[j] = - points[j, 1].copy()
      self.xmax = max(x[j], self.xmax)
      self.ymax = max(y[j], self.ymax)
      self.zmax = max(z[j], self.zmax)
      self.xmin = min(x[j], self.xmin)
      self.ymin = min(y[j], self.ymin)
      self.zmin = min(z[j], self.zmin)
    if c == 'auto':
      c = [(z[j] + 0.5, y[j] + 0.5, x[j] + 0.5) for j in range(points.shape[0])]
    self.ax.scatter(x, y, z, s = 200, c = c, marker = marker)
    for e in edges:
      self.ax.plot(x[e], y[e], z[e], c = c)
    
  def show_3d(self):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.show()
    
  def add_img(self, img, imgId = 'default'):
    self.imgs[imgId] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(mask.shape[0], mask.shape[1], 1) * 255 * trans +                         bg * (1 - trans)).astype(np.uint8)

  def add_point_2d(self, point, c, imgId='default'):
    self.imgs[imgId] = show_2d(self.imgs[imgId], point, c, self.edges)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def show_all_imgs(self, pause = False):
    if not self.ipynb:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if pause:
        cv2.waitKey()
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          plt.imshow(v)
      plt.show()
  
  def save_3d(self, path):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.savefig(path, bbox_inches='tight', frameon = False)
  
  def save_img(self, imgId = 'default', path = '../debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path = '../debug/'):
    for i, v in self.imgs.items():
      cv2.imwrite(path + '/{}.png'.format(i), v)


# ## Setup the Model

# In[ ]:


sys.argv=[''] # get rid of silly jupyter cmdline args
opt = opts().parse()


# In[ ]:


opt.heads['depth'] = opt.num_output
opt.load_model = os.path.join(POSE_ROOT, 'models', 'fusion_3d_var.pth')
opt.device = torch.device('cuda:0')
model, _, _ = create_model(opt)
model = model.to(opt.device)
model.eval()


# In[ ]:


import os
if False:
  from google.colab.drive import mount as mount_gdrive
  from google.colab.files import download as FileLink
  torch.save(model, 'full_model.pth')
  gd_path = os.path.join(os.getcwd(), 'gdrive')
  mount_gdrive(gd_path)
  get_ipython().system('cp full_model.pth /content/gdrive/My\\ Drive/')


# ## Process Output
# Here we have the code to process and analyze the output

# In[ ]:


from scipy.ndimage import zoom
def run_predictions(image, model, opt, show_image=False):
  s = max(image.shape[0], image.shape[1]) * 1.0
  c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
  trans_input = get_affine_transform(
      c, s, 0, [opt.input_w, opt.input_h])
  inp = cv2.warpAffine(image, trans_input, (opt.input_w, opt.input_h),
                         flags=cv2.INTER_LINEAR)
  inp = (inp / 255. - mean) / std
  inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
  inp = torch.from_numpy(inp).to(opt.device)
  out = model(inp)[-1]
  pred_raw = get_preds(out['hm'].detach().cpu().numpy())[0]
  
  pred = transform_preds(pred_raw, c, s, (opt.output_w, opt.output_h))
  
  depth = out['depth'].detach().cpu().numpy()
  pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         depth)[0]
  
  joint_prob_map = out['hm'].detach().cpu().numpy()[0]
  local_joint_confidence = np.max(joint_prob_map, (1, 2)).reshape((-1, 1))
  
  region_joint_confidence = np.max(zoom(joint_prob_map, [1, 0.5, 0.5], order=2), 
                                   (1, 2)).reshape((-1, 1))
  pred = np.concatenate([pred, local_joint_confidence, region_joint_confidence], 1)
  pred_3d = np.concatenate([pred_3d, local_joint_confidence, region_joint_confidence], 1)
  
  return pred, pred_3d, (depth[0], joint_prob_map)


# In[ ]:


model.to(opt.device);


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred_2d, pred_3d, (depth, pred_map) = run_predictions(frame, model, opt)')


# In[ ]:


try:
    from skimage.util.montage import montage2d
except:
    from skimage.util import montage as montage2d
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 5))
ax1.imshow(frame)
ax1.set_title('Input Image')
ax2.imshow(montage2d(depth))
ax2.set_title('Depth Map Montage')
ax3.imshow(np.sum(pred_map, 0))
ax3.set_title('Pose Confidence')
ax4.imshow(np.argmax(pred_map, 0)*(np.sum(pred_map, 0)>0.5), cmap='tab20', interpolation='none')
ax4.set_title('Confident Pose Spots')


# In[ ]:


debugger = Debugger(ipynb=True)
debugger.add_img(frame)
debugger.add_point_2d(pred_2d, (255, 0, 0))
debugger.show_all_imgs(pause=False)


# In[ ]:


crop_frame = frame[::2, 300:1380:2]
crop_frame = np.clip(crop_frame*255.0/np.percentile(frame.max(), 80), 
                           0, 255).astype('uint8')
pred_2d, pred_3d, (depth, pred_map) = run_predictions(crop_frame, model, opt)

debugger = Debugger(ipynb=True)
debugger.add_img(crop_frame)
debugger.add_point_2d(pred_2d[:, :2], (255, 0, 0))
debugger.add_point_3d(pred_3d[:, :3], 'b')
debugger.show_all_imgs(pause=False)


# In[ ]:


def pred_maps(pred_map, depth):
  """
  create nice intermediate representations of the predictions
  """
  joint_conf = np.sum(pred_map, 0)
  joint_id_map = np.argmax(pred_map, 0)

  _, xx, yy = np.meshgrid(1, 
                          np.arange(pred_map.shape[1]), 
                          np.arange(pred_map.shape[2]), 
                         indexing='ij')

  joint_depth = depth[joint_id_map[xx, yy].ravel(), xx.ravel(), yy.ravel()].reshape(xx.shape[1:])
  return joint_conf, joint_id_map, joint_depth


# In[ ]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 5))
ax1.imshow(crop_frame[:, :, ::-1])
ax1.set_title('Input Image')

joint_conf, joint_id_map, joint_depth = pred_maps(pred_map, depth)
ax2.imshow(joint_depth*(joint_conf>0.2), cmap='RdBu')
ax2.set_title('Depth Map')
ax3.imshow(joint_conf)
ax3.set_title('Pose Confidence')
ax4.imshow(joint_id_map*(joint_conf>0.25), 
           cmap='tab20', interpolation='none')
ax4.set_title('Confident Pose Spots')


# In[ ]:


from PIL import Image
def make_grid(frame, depth, pred_map):
  joint_conf, joint_id_map, joint_depth = pred_maps(pred_map, depth)
  joint_depth[joint_conf<0.1] = joint_depth.min()
  # make lil thumbnail iamges
  x_dim, y_dim = depth.shape[1:]
  thumb_img = np.array(Image.fromarray(frame[:, :, ::-1]).resize((x_dim*2, y_dim*2)))
  conf_img = plt.cm.viridis(joint_conf)[:, :, :3]*255
  
  jp_img = plt.cm.gray(joint_depth+0.5)[:, :, :3]*255
  max_depth_img = plt.cm.RdBu(np.median(depth,0)+0.5)[:, :, :3]*255
  js_img = plt.cm.tab20(joint_id_map*(joint_conf>0.25))[:, :, :3]*255
  return np.clip(np.hstack([thumb_img, 
                            np.vstack([conf_img,js_img]),
                            np.vstack([jp_img, max_depth_img])]),
                 0, 255).astype('uint8')
  

grid_thumb = make_grid(crop_frame, depth, pred_map)
plt.imshow(grid_thumb)


# ## Export Videos

# ### Old Video Processing Code (OpenCV)

# In[ ]:


import pandas as pd
def process_video_cv(in_file, 
                  out_file, 
                  out_grid_file,
                  speed_factor=0.5,
                  skip_frames=2, 
                  max_frames=100):
  
  c_cap = cv2.VideoCapture(in_file)
  out_df = []
  # dummy read the first frame
  _, frame = c_cap.read() 
  frame_cropper = lambda x: x
  crop_frame = frame_cropper(frame)
  frame_width, frame_height, _ = crop_frame.shape
  pred_2d, pred_3d, (depth, pred_map) = run_predictions(crop_frame, 
                                                            model, opt)
  
  jnt_df = pd.DataFrame(pred_3d, columns=["x", "y", "z", "peak_conf", "reg_conf"])
  jnt_df['pos_msec'] = c_cap.get(cv2.CAP_PROP_POS_MSEC)
  out_df += [jnt_df.rename_axis("joint").reset_index()]
  grid_thumb = make_grid(crop_frame, depth, pred_map)
  frame_rate = int(speed_factor*c_cap.get(cv2.CAP_PROP_FPS)/skip_frames)
  frame_rate = 30
  if out_file is not None:
    out = cv2.VideoWriter(out_file,
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          frame_rate, 
                          (frame_height, frame_width))
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
  if out_grid_file is not None:
    out_grid = cv2.VideoWriter(out_grid_file,
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          frame_rate, 
                          (grid_thumb.shape[1], grid_thumb.shape[0]))
    out_grid.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
  
  frame_count = 0
  while(c_cap.isOpened()):
    for i in range(skip_frames):
      ret, frame = c_cap.read()
    if ret == True:
      crop_frame = frame_cropper(frame)
      # auto brighten
      crop_frame = np.clip(crop_frame*255.0/np.percentile(frame.max(), 60), 
                           0, 255).astype('uint8')
      pred_2d, pred_3d, (depth, pred_map) = run_predictions(crop_frame, 
                                                            model, opt)
      # opencv is a wimp
      crop_frame = np.ascontiguousarray(crop_frame, dtype=np.uint8)
      out_img = show_2d(crop_frame, pred_2d, (255, 0, 0), mpii_edges)
      out_img = np.ascontiguousarray(out_img, dtype=np.uint8)
      if out_file is not None:
        out.write(out_img)
      grid_thumb = make_grid(crop_frame, depth, pred_map)
      # RGB -> BGR
      grid_thumb = np.ascontiguousarray(grid_thumb[:, :, ::-1], dtype=np.uint8)
      if out_grid_file is not None:
        out_grid.write(grid_thumb)
      frame_count+=1
      if max_frames<50:
        plt.imshow(grid_thumb)
    if not ret or (frame_count>=max_frames):
      break
  
  out_csv_name = '{}.csv'.format(out_file)
  pd.concat(out_df).to_csv(out_csv_name, index=False)
  c_cap.release()
  if out_file is not None:
    out.release()
  if out_grid_file is not None:
    out_grid.release()


# ### Save and Export Everything

# In[ ]:


for i, c_video in enumerate(local_videos):
    out_video = os.path.split(c_video)[1]+'_out.avi',
    grid_video = os.path.split(c_video)[1]+'_grid.avi'
    process_video_cv(c_video, out_video, grid_video,
                max_frames=100, skip_frames=1, speed_factor=0.65)
    


# In[ ]:


get_ipython().system('ls -lh *.mp4')


# In[ ]:


get_ipython().system('tar -czf pose_3d pose_3d.tar.gz')


# In[ ]:


get_ipython().run_cell_magic('sh', '', 'rm -rf pose_3d')


# In[ ]:


out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (grid_thumb.shape[1], grid_thumb.shape[0]))


# In[ ]:


out.


# In[ ]:




