#!/usr/bin/env python
# coding: utf-8

# # Trainable Image Resize
# 
# It was mentioned in the forums https://www.kaggle.com/c/data-science-bowl-2018/discussion/52766 that the fact that images are at different scales can be a significant problem. 
# It was also pointed that people dealt with it before in the following manner:
# 
# * train network on raw images
# 
# * run image through the net
# 
# * estimate object size
# 
# * resize based on the object size estimate
# 
# * train new network on resize images
# 
# We decided to implement it in the following way:
# 
# * train unet for mask and contour prediction
# 
# * do some morphological postprocessing mask+contour
# 
# * estimage size by:
# 
# ```python
# class CellSizer(BaseTransformer):
#     def __init__(self, **kwargs):
#         pass
# 
#     def transform(self, labeled_images):
#         mean_sizes = []
#         for image in tqdm(labeled_images):
#             mean_size = mean_cell_size(image)
#             mean_sizes.append(mean_size)
#         return {'sizes': mean_sizes}
#             
# def mean_cell_size(labeled_image):
#     blob_sizes = itemfreq(labeled_image)
#     if blob_sizes.shape[0]==1:
#         return 0
#     else:
#         blob_sizes = blob_sizes[blob_sizes[:, 0].argsort()][1:, 1]
#         return np.mean(blob_sizes)
#             
# ```
# 
# * rescaling the image (assuming certain boundaries) with
# 
# ```python 
# 
# class ImageReaderRescaler(BaseTransformer):
#     def __init__(self, min_size, max_size, target_ratio):
#         self.min_size = min_size
#         self.max_size = max_size
#         self.target_ratio = target_ratio
# 
#     def _transform(self, sizes, X, y=None, meta=None):
#         raw_images = X[0]
#         raw_images_adj = []
#         for size, raw_image in tqdm(zip(sizes, raw_images)):
#             h_adj, w_adj = self._get_adjusted_image_size(size, from_pil(raw_image))
#             raw_image_adj = resize(from_pil(raw_image), (h_adj, w_adj), 
#                                    preserve_range=True).astype(np.uint8)
#             raw_images_adj.append(to_pil(raw_image_adj))
#         X_adj = [raw_images_adj]
#         ...
#         return X_adj, y_adj
# 
#     def _get_adjusted_image_size(self, mean_cell_size, img):
#         h, w = img.shape[:2]
#         img_area = h * w
#         
#         if mean_cell_size ==0:
#             adj_ratio = 1.0
#         else:
#             size_ratio = img_area / mean_cell_size
#             adj_ratio = size_ratio / self.target_ratio
# 
#         h_adj = int(clip(self.min_size, h * adj_ratio, self.max_size))
#         w_adj = int(clip(self.min_size, w * adj_ratio, self.max_size))
# 
#         return h_adj, w_adj
# ```
# 
# * Finally on such rescaled images we train and predict by using patches of fixed size (say 512x512). For example inference can be done with something like this:
# 
# ```python
# class PatchCombiner(BaseTransformer):
# ```
#     ...
# ```python
#     def _join_output(self, patch_meta, image_patches):
#         image_h = patch_meta['image_h'].unique()[0]
#         image_w = patch_meta['image_w'].unique()[0]
#         prediction_image = np.zeros((image_h, image_w))
#         prediction_image_padded = get_mosaic_padded_image(prediction_image, 
#                                                           self.patching_size, 
#                                                           self.patching_stride)
# 
#         patches_per_image = 0
#         for (y_coordinate, 
#              x_coordinate, 
#              tta_angle), image_patch in zip(patch_meta[['y_coordinates', 
#                                                         'x_coordinates', 
#                                                         'tta_angles']].values.tolist(), 
#                                             image_patches):
#             patches_per_image += 1
#             image_patch = np.rot90(image_patch, -1 * tta_angle / 90.)
#             (window_y, 
#              window_x) = y_coordinate * self.patching_stride, x_coordinate * self.patching_stride
#             prediction_image_padded[window_y:self.patching_size + window_y,
#             window_x:self.patching_size + window_x] += image_patch
# 
#         _, h_top, h_bottom, _ = get_padded_size(max(image_h, self.patching_size),
#                                                 self.patching_size,
#                                                 self.patching_stride)
#         _, w_left, w_right, _ = get_padded_size(max(image_w, self.patching_size),
#                                                 self.patching_size,
#                                                 self.patching_stride)
# 
#         prediction_image = prediction_image_padded[h_top:-h_bottom, w_left:-w_right]
#         prediction_image /= self.normalization_factor
#         return prediction_image
# ```
# 

# # Full pipeline
# If you would like to see how we plugged trainable rescale into our pipeline go to [open solution](https://github.com/neptune-ml/open-solution-data-science-bowl-2018)
# 
# ![full open solution pipeline](https://gist.githubusercontent.com/jakubczakon/10e5eb3d5024cc30cdb056d5acd3d92f/raw/e85c1da3acfe96123d0ff16f8145913ee65e938c/full_pipeline.png)
# 
# The `ImageReaderRescaler` step is defined in the `preprocessing.py` file:
# 
# If you want to use our implementation just go for it!

# In[ ]:




