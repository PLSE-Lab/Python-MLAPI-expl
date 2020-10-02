#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import os


def normalize_image(img, org_width, org_height, new_width, new_height):
  # Invert
  img = 255 - img
  # Normalize
  img = (img * (255.0 / img.max())).astype(np.uint8)
  # Reshape
  img = img.reshape(org_height, org_width)
  image_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
  return image_resized


def dump_images(args, org_width, org_height, new_width, new_height):
  labels = pd.read_csv(args.labels)
  iids = labels['image_id']
  root = labels['grapheme_root']
  vowel = labels['vowel_diacritic']
  consonant = labels['consonant_diacritic']
  labels = {a: (b, c, d) for a, b, c, d in zip(iids, root, vowel, consonant)}
  tuples = sorted(set(labels.values()))
  tuples_to_int = {v: k for k, v in enumerate(tuples)}
  print(f'Got {len(tuples)} unique combinations')
  for i in tqdm(range(0, 4)):
    df = pd.read_parquet(args.data_template % i)
    image_ids = df['image_id'].values
    df = df.drop(['image_id'], axis=1)
    for image_id, index in tqdm(zip(image_ids, range(df.shape[0])), total=df.shape[0]):
      normalized = normalize_image(df.loc[df.index[index]].values,
          org_width, org_height, new_width, new_height)
      r, v, c = labels[image_id]
      tuple_int = tuples_to_int[(r, v, c)]
      # e.g: 'Train_300_rt_29_vl_5_ct_0_ti_179.png'
      out_fn = os.path.join(args.image_dir, f'{image_id}_rt_{r}_vl_{v}_ct_{c}_ti_{tuple_int}.png')
      cv2.imwrite(out_fn, normalized)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir', type=str, default='images')
  parser.add_argument('--data_template', type=str, default='../input/bengaliai-cv19/train_image_data_%d.parquet')
  parser.add_argument('--labels', type=str, default='../input/bengaliai-cv19/train.csv')
  args, _ = parser.parse_known_args()

  os.makedirs(args.image_dir, exist_ok=True)

  org_height = 137
  org_width = 236
  new_height = 160  # 5 * 32
  new_width = 256  # 8 * 32
  dump_images(args, org_width, org_height, new_width, new_height)
  print(f'Done wrote to {args.image_dir}')

main()


# In[ ]:


# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
# author: Martin Gorner
# twitter: @martin_gorner
# modified: See--
# modified from:
# https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/03_Flower_pictures_to_TFRecords.ipynb
"""
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor


def read_image_label(inputs):
  img_bytes = tf.io.read_file(inputs['img'])
  return img_bytes, inputs['image_id'], inputs['grapheme_root'], inputs['vowel_diacritic'],       inputs['consonant_diacritic'], inputs['unique_tuple']


def to_tfrecord(img_bytes, image_id, grapheme_root, vowel_diacritic,
      consonant_diacritic, unique_tuple):
  feature = {
      'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
      'image_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_id])),
      'grapheme_root': tf.train.Feature(int64_list=tf.train.Int64List(value=[grapheme_root])),
      'vowel_diacritic': tf.train.Feature(int64_list=tf.train.Int64List(value=[vowel_diacritic])),
      'consonant_diacritic': tf.train.Feature(int64_list=tf.train.Int64List(value=[
          consonant_diacritic])),
      'unique_tuple': tf.train.Feature(int64_list=tf.train.Int64List(value=[unique_tuple])),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def get_img_size(fn):
  try:
    # width, height = im.size
    img_size = Image.open(fn).size[::-1]

  except Exception as e:
    print(f'{fn} errored with {e}')
    img_size = None
  return img_size


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--clean', action='store_true')
  parser.add_argument('--version', type=str, default='v0.1.0')
  parser.add_argument('--do_not_train', action='store_true')
  parser.add_argument('--records_dir', type=str, default='records')
  parser.add_argument('--image_glob', type=str, default='images/*.png')
  parser.add_argument('--seed', type=int, default=123)
  args, _ = parser.parse_known_args()

  np.random.seed(args.seed)
  os.makedirs(args.records_dir, exist_ok=True)
  if args.clean:
    os.system(f'rm -f {args.records_dir}/*.tfrec')
    print('Done cleaning')
    return 0

  fns = sorted(tf.io.gfile.glob(args.image_glob),
      key=lambda x: int(x.split('_')[1]))
  perm = np.random.permutation(len(fns))
  vaild_len = int(0.1 * len(fns))
  val_fns = [fns[p] for p in perm[:vaild_len]]
  print("val_len:{}".format(len(perm[:vaild_len])))
  index = [i for i in range(0,vaild_len)]
  perm = np.delete(perm,index)
  print("train_len:{}".format(len(perm)))
  train_fns = [fns[p] for p in perm]
  
  print(f'{len(train_fns)} training and {len(val_fns)} validation fns')
  num_shards = 1
  for prefix in ['val', 'train']:
    if prefix == 'train' and args.do_not_train:
      continue

    if prefix == 'train':
      img_filenames = train_fns
    else:
      img_filenames = val_fns

    print('Removing images with bad shape')
    # remove images with bad shape
    with ThreadPoolExecutor() as e:
      img_sizes = list(tqdm(e.map(get_img_size, img_filenames), total=len(
        img_filenames)))

    img_sizes = [tf.constant(sz, tf.int64) for sz in img_sizes]

    # e.g: 'images/Train_116991_rt_53_vl_7_ct_4_ti_343.png'
    #       000000000000_111111_22_33_44_5_66_7_88_9999999
    image_id = [int(fn.split('_')[1]) for fn in img_filenames]
    grapheme_root = [int(fn.split('_')[3]) for fn in img_filenames]
    vowel_diacritic = [int(fn.split('_')[5]) for fn in img_filenames]
    consonant_diacritic = [int(fn.split('_')[7]) for fn in img_filenames]
    unique_tuple = [int(fn.split('_')[9][:-4]) for fn in img_filenames]

    if prefix == 'train':
      num_shards = 10
    else:
      num_shards = 2

    ds = tf.data.Dataset.from_tensor_slices({'img': img_filenames, 'image_id': image_id,
        'grapheme_root': grapheme_root, 'vowel_diacritic': vowel_diacritic,
        'consonant_diacritic': consonant_diacritic, 'unique_tuple': unique_tuple})
    ds = ds.map(read_image_label)
    ds = ds.batch(len(img_filenames) // num_shards)
    print("Writing TFRecords")
    for shard_index, ret in tqdm(enumerate(ds), total=num_shards):
      # batch size used as shard size here
      img, image_id, r, v, c, ti = map(lambda x: x.numpy(), ret)
      current_shard_size = img.shape[0]
      # good practice to have the number of records in the filename
      filename = os.path.join(args.records_dir, 'HALF_%s_%04d_%06d_%s.tfrec' % (
          prefix, shard_index, current_shard_size, args.version))
      with tf.io.TFRecordWriter(filename) as out_file:
        for i in tqdm(range(current_shard_size)):
          example = to_tfrecord(img[i], image_id[i], r[i], v[i], c[i], ti[i])
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, current_shard_size))


main()


# In[ ]:


get_ipython().system('du -sh images')


# In[ ]:


get_ipython().system('rm -rf images')


# In[ ]:


get_ipython().system('ls records')


# In[ ]:




