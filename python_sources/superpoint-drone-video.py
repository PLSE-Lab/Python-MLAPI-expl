#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook shows how to apply the pretrained SuperPoint network to drone videos for tracking movement and ultimately reconstructing scene and position information. 
# 
# ### Part 2
# The second part is about understanding the model and converting it into a more generic format (Keras, although ONNX would be fine too) so we can reimplement it on other platforms without having Torch present. Keras also means we could compile it to run in a browser environment (keras.js and tensorflow.js) although that might require converting a lot of opencv functions as well.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import numpy as np
import os
import time

import cv2
import torch
# Jet colormap for visualization.
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


# # SuperPointNet 
# Here is the basic code for the pytorch model where we load the pretrained weights to identfy 'superpoints'

# In[ ]:


class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc


# # Configure Settings
# Here we specify the weights, the video to process and the size of the analysis to run

# In[ ]:


weights_path = '../input/superpoint-pretrained-network/superpointpretrainednetwork-master/SuperPointPretrainedNetwork-master/superpoint_v1.pth'
video_path = '../input/drone-videos/Stockflue Flyaround.mp4'


# In[ ]:


img_w, img_h = 240, 320
img_scale = 3
max_frame_count = 25


# # Parse the Video

# In[ ]:


class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image_bgr = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image_bgr, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, input_image_bgr, True)


# In[ ]:


vs = VideoStreamer(video_path, 0, img_w, img_h, skip=4, img_glob='*.mp4')


# In[ ]:


# show a preview frame
plt.imshow(vs.next_frame()[1][:, :, ::-1])


# # Load Pretrained Network

# In[ ]:


class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # Load the network in inference mode.
    self.net = SuperPointNet()
    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap


# In[ ]:


# This class runs the SuperPoint network and processes its outputs.
fe = SuperPointFrontend(weights_path=weights_path,
                      nms_dist=4,
                      conf_thresh=0.015,
                      nn_thresh=0.7,
                      cuda=0)


# # Use Point Tracker to Combine Tracks

# In[ ]:


class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.
  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.
    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.
    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.
    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.
    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


# In[ ]:


# This class helps merge consecutive point matches into tracks.
tracker = PointTracker(10, nn_thresh=fe.nn_thresh)


# In[ ]:


# Font parameters for visualizaton.
font = cv2.FONT_HERSHEY_DUPLEX
font_clr = (255, 255, 255)
font_pt = (4, 12)
font_sc = 0.4


# # Run the Analysis

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize = (12, 4))
while True:
    start = time.time()

    # Get a new image.
    img, img_bgr, status = vs.next_frame()
    if status is False:
        break

    # Get points and descriptors.
    start1 = time.time()
    pts, desc, heatmap = fe.run(img)
    end1 = time.time()

    # Add points and descriptors to the tracker.
    tracker.update(pts, desc)

    # Get tracks for points which were match successfully across all frames.
    tracks = tracker.get_tracks(2)

    # Primary output - Show point tracks overlayed on top of input image.
    out1 = img_bgr # use a color image
    tracks[:, 1] /= float(fe.nn_thresh) # Normalize track scores to [0,1].
    tracker.draw_tracks(out1, tracks)
    cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show current point detections.
    out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
    cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show the point confidence heatmap.
    if heatmap is not None:
        min_conf = 0.001
        heatmap[heatmap < min_conf] = min_conf
        heatmap = -np.log(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
        out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
        out3 = (out3*255).astype('uint8')
    else:
        out3 = np.zeros_like(out2)
    cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

    # Resize final output.
    out = np.hstack((out1, out2, out3))
    out = cv2.resize(out, (3*img_scale*img_w, img_scale*img_h))

    # Display visualization image to screen.
    clear_output()
    ax1.imshow(out[:, :, ::-1])
    display(fig)

    out_file = 'frame_%05d.png' % vs.i
    cv2.imwrite(out_file, out)

    end = time.time()
    net_t = (1./ float(end1 - start))
    total_t = (1./ float(end - start))
    print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).'        % (vs.i, net_t, total_t))
    if vs.i>max_frame_count:
        break


# # Make Animated GIFs

# In[ ]:


get_ipython().system('ls -lh * | head -n 10')


# In[ ]:


import matplotlib.animation as animation
Writer = animation.writers['imagemagick']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

from skimage.io import imread
all_pngs = sorted(glob.glob('*.png'))


# In[ ]:


from tqdm import tqdm

ims = []
plt.close('all')
fig, ax1 = plt.subplots(1,1, figsize = (3*img_w/10, img_h/10))
f_img = imread(all_pngs[0])
c_aximg = ax1.imshow(f_img, animated = True)
ax1.axis('off')
plt.tight_layout()
def update_image(frame):
    if frame>0:
        r_img = imread(all_pngs[frame])
    else:
        r_img = f_img
    c_aximg.set_array(r_img)
    return c_aximg,

im_ani = animation.FuncAnimation(fig, 
                                 update_image, 
                                 frames = range(len(all_pngs)),
                                 interval=200, repeat_delay=300,
                                blit=True)
im_ani.save('out.gif', writer=writer)


# In[ ]:


rm *.png


# # Extract Weights
# If we extract the weights we can make a Keras model out of it. Keras models (and HDF5) are much more stable fileformats that are easy to read in a number of different tools. PyTorch is pretty primitive at the moment using pickles for everything and until version 1.0 it is unlikely that the format itself will have any stability.  Furthermore making a Keras model lets us easily switch between Tensorflow, Theano, CNTK and MXNet as a backend based on whatever platform provides the most useful features.

# In[ ]:


print(fe.net)


# In[ ]:


# Save weights to disk
import h5py
net_state = fe.net.state_dict()
with h5py.File('model_weights.h5', 'w') as hf:
    for k in net_state.keys():
        hf.create_dataset(k, data = net_state[k].numpy())
    for k in hf.keys():
        print(k, hf[k].shape)


# # Create Keras Model
# We can use the pytorch code as the basis for a keras model
# ## Model
# ```
#  self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
# self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
# self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
# self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
# self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
# self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
# self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
# self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
# # Detector Head.
# self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
# self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
# # Descriptor Head.
# self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
# self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
# ```
# ## Forward
# ```
# x = self.relu(self.conv1a(x))
# x = self.relu(self.conv1b(x))
# x = self.pool(x)
# x = self.relu(self.conv2a(x))
# x = self.relu(self.conv2b(x))
# x = self.pool(x)
# x = self.relu(self.conv3a(x))
# x = self.relu(self.conv3b(x))
# x = self.pool(x)
# x = self.relu(self.conv4a(x))
# x = self.relu(self.conv4b(x))
# # Detector Head.
# cPa = self.relu(self.convPa(x))
# semi = self.convPb(cPa)
# # Descriptor Head.
# cDa = self.relu(self.convDa(x))
# desc = self.convDb(cDa)
# dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
# desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
#  ```

# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Lambda # we might need lambda for some of the normalization stuffs
import keras.backend as K
def pool():
    return MaxPool2D(strides=2, pool_size=(2,2))
def conv_from_key(k, padding = 1, activation = 'relu'):
    w = net_state['{}.weight'.format(k)].numpy()
    b = net_state['{}.bias'.format(k)].numpy()
    filt_num, in_dim, x_wid, y_wid = w.shape
    
    w_keras = w.swapaxes(0, 3).swapaxes(1, 2)
    w_keras = w_keras.swapaxes(0, 1) # try switching x and y
    
    return Conv2D(filt_num, kernel_size=(x_wid, y_wid), weights = [w_keras, b], name = k, padding ='same' if padding else 'valid', activation = activation)


# In[ ]:


img_in = Input((None, None , 1)) # keep the model flexible (if unreadable)
conv1a = conv_from_key('conv1a')(img_in)
conv1b = conv_from_key('conv1b')(conv1a)
pool1 = pool()(conv1b)

conv2a = conv_from_key('conv2a')(pool1)
conv2b = conv_from_key('conv2b')(conv2a)
pool2 = pool()(conv2b)

conv3a = conv_from_key('conv3a')(pool2)
conv3b = conv_from_key('conv3b')(conv3a)
pool3 = pool()(conv3b)

conv4a = conv_from_key('conv4a')(pool3)
conv4b = conv_from_key('conv4b')(conv4a)

convPa = conv_from_key('convPa')(conv4b)
semi = conv_from_key('convPb', padding = 0, activation = 'linear')(convPa)

convDa = conv_from_key('convDa')(conv4b)
desc = conv_from_key('convDb', padding = 0, activation = 'linear')(convDa)


# ## Solve Normalization
# The function is running over the first value (depth in torch and thus the last dimension in keras), we want to make sure the normalization we use gives the same results as the torch ones
# ```
# f = np.array([[ 2.1983,  0.4141],
#             [ 0.8734,  1.9710],
#             [-0.7778,  0.7938],
#             [-0.1342,  0.7347]])
#             
# torch.norm(f, 2, 1) = tensor([ 2.2369,  2.1558,  1.1113,  0.7469])
# ```

# In[ ]:


f = np.array([[ 2.1983,  0.4141],
            [ 0.8734,  1.9710],
            [-0.7778,  0.7938],
            [-0.1342,  0.7347]])
np.sqrt(np.sum(np.square(f), 1))


# In[ ]:


torch_norm_layer = Lambda(lambda x: K.tile(K.sqrt(K.sum(K.square(x), 3, keepdims = True)), [1, 1, 1, K.shape(x)[3]]), name = 'NormLayer')
norm_out = torch_norm_layer(desc)
desc_out = Lambda(lambda x: x[0]/x[1],name = 'DivideLayer')([desc, norm_out])


# In[ ]:


sp_model = Model(inputs = [img_in], 
                outputs = [semi, desc_out])
sp_model.save('superpointnet_keras.h5')
sp_model.summary()


# # Compare Torch to Keras
# This step is critical and should ideally be done on  a few different images. Here we just take a single sample image and compare the outputs pixel-by-pixel for each channel to ensure they are as close as possible (they will never be exactly equal since these models are quite complex). We then show the differences visually to see if any particular issues arise like at borders (padding can be very different).  We finally plot the one output on a linear axis against the other in case there are weird cropping or saturation effects.

# In[ ]:


H, W = img.shape[0], img.shape[1]
inp = img.copy()
inp = (inp.reshape(1, H, W))
inp = torch.from_numpy(inp)
inp = torch.autograd.Variable(inp).view(1, 1, H, W)
# Forward pass of network.
outs = fe.net.forward(inp)
semi_out, desc_out = outs[0].cpu().detach().numpy(), outs[1].cpu().detach().numpy()

print(semi_out.shape, desc_out.shape)


# In[ ]:


ksemi_out, kdesc_out = sp_model.predict(np.expand_dims(np.expand_dims(img,0), -1))
print(ksemi_out.shape, kdesc_out.shape)


# ## Compare Outputs

# In[ ]:


t_to_k = lambda x: x.swapaxes(1, 3).swapaxes(2, 3)
print('Semi:', 'Mean Abs Torch Output', np.mean(np.abs(semi_out)), 'Mean Abs Diff Output', np.mean(np.abs(t_to_k(ksemi_out)-semi_out)))
print('Desc:', 'Mean Abs Torch Output', np.mean(np.abs(desc_out)), 'Mean Abs Keras Output', np.mean(np.abs(kdesc_out)), 'Mean Abs Diff Output', np.mean(np.abs(t_to_k(kdesc_out)-desc_out)))


# In[ ]:


from skimage.util.montage import montage2d
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
pltargs = dict(cmap = 'RdBu', vmin = -10, vmax = 10)
ax1.imshow(montage2d(semi_out[0]), **pltargs)
ax1.set_title('Semi Torch Output')
ax2.imshow(montage2d(t_to_k(ksemi_out)[0]), **pltargs)
ax2.set_title('Semi Keras Output')
ax3.imshow(montage2d(semi_out[0]-t_to_k(ksemi_out)[0]), **pltargs)
ax3.set_title('Semi Diff Output')


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
pltargs = dict(cmap = 'RdBu', vmin = -0.5, vmax = 0.5)
ax1.imshow(montage2d(desc_out[0]), **pltargs)
ax1.set_title('Desc Torch Output')
ax2.imshow(montage2d(t_to_k(kdesc_out)[0]), **pltargs)
ax2.set_title('Desc Keras Output')
ax3.imshow(montage2d(desc_out[0]-t_to_k(kdesc_out)[0]), **pltargs)
ax3.set_title('Desc Diff Output')


# In[ ]:


fig, (ax2, ax1) = plt.subplots(1, 2, figsize = (20, 10))
ax1.plot(desc_out.ravel(), t_to_k(kdesc_out).ravel(), '.')
ax1.set_title('Desc Plot')
ax2.plot(semi_out.ravel(), t_to_k(ksemi_out).ravel(), '.')
ax2.set_title('Semi Plot')


# ## Result
# The networks appear to be very closely matched so we can proceed with the keras model. There is still some work to do with the `grid_sampling` code from torch since we will likely do this in scipy and small differences might appear.

# In[ ]:




