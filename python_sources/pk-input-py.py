# Author: xiaotaw@qq.com (Any bug report is welcome)
# Time: Aug 2016
# Addr: Shenzhen
# Description: define functions and parameters related to input data


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import time
import random

import numpy as np

from scipy import sparse

vec_len = 2039
h5_dir = "../input"



def dense_to_one_hot(labels_dense, num_classes=2, dtype=np.int):
  """Convert class labels from scalars to one-hot vectors.
  Args:
    labels_dense: <type 'numpy.ndarray'> dense label
    num_classes: <type 'int'> the number of classes in one hot label
    dtype: <type 'type'> data type
  Return:
    labels_ont_hot: <type 'numpy.ndarray'> one hot label
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel().astype(dtype)] = 1
  return labels_one_hot


class Dataset(object):
  """Base dataset class 
  """
  def __init__(self, size, is_shuffle=False, fold=10):
    """Constructor, create a dataset container. 
    Args:
      size: <type 'int'> the number of samples
      is_shuffle: <type 'bool'> whether shuffle samples when the dataset created
      fold: <type 'int'> how many folds to split samples
    Return:
      None
    """
    self.size = size
    self.perm = np.array(range(self.size))
    if is_shuffle:
      random.shuffle(self.perm)
 
    self.train_size = int(self.size * (1.0 - 1.0 / fold))
    self.train_perm = self.perm[range(self.train_size)]
    self.train_begin = 0
    self.train_end = 0

    self.test_perm = self.perm[range(self.train_size, self.size)]

  def generate_perm_for_train_batch(self, batch_size):
    """Create the permutation for a batch of train samples
    Args:
      batch_size: <type 'int'> the number of samples in the batch
    Return:
      perm: <type 'numpy.ndarray'> the permutation of samples which form a batch
    """
    self.train_begin = self.train_end
    self.train_end += batch_size
    if self.train_end > self.train_size:
      random.shuffle(self.train_perm)
      self.train_begin = 0
      self.train_end = batch_size
    perm = self.train_perm[self.train_begin: self.train_end]
    return perm


class PosDataset(Dataset):
  """Positive dataset class
  """
  def __init__(self, target, one_hot=True, dtype=np.float32):
    """Create a positive dataset for a protein kinase target.
      The data is read from hdf5 files.
    Args:
      target: <type 'str'> the protein kinase target name, also the name of hdf5 file
      one_hot: <type 'bool'> whether to convert labels from dense to one_hot
      dtype: <type 'type'> data type of features 
    Return:
      None
    """
    # open h5 file
    self.h5_fn = os.path.join(h5_dir, target + ".h5")
    self.h5 = h5py.File(self.h5_fn, "r")
    # read ids
    self.ids = self.h5["chembl_id"].value
    # read 3 fp, and stack as feauture
    ap = sparse.csr_matrix((self.h5["ap"]["data"], self.h5["ap"]["indices"], self.h5["ap"]["indptr"]), shape=[len(self.h5["ap"]["indptr"]) - 1, vec_len])
    mg = sparse.csr_matrix((self.h5["mg"]["data"], self.h5["mg"]["indices"], self.h5["mg"]["indptr"]), shape=[len(self.h5["mg"]["indptr"]) - 1, vec_len])
    tt = sparse.csr_matrix((self.h5["tt"]["data"], self.h5["tt"]["indices"], self.h5["tt"]["indptr"]), shape=[len(self.h5["tt"]["indptr"]) - 1, vec_len])
    self.features = sparse.hstack([ap, mg, tt]).toarray()
    # label 
    self.labels = self.h5["label"].value
    if one_hot == True:
      self.labels = dense_to_one_hot(self.labels)
    # year
    if "year" in self.h5.keys():
      self.years = self.h5["year"].value
    else:
      self.years = None
    # close h5 file
    self.h5.close()
    # dtype
    self.dtype = dtype
    # pre_process
    #self.features = np.log10(1.0 + self.features).astype(self.dtype)
    self.features = np.clip(self.features, 0, 1).astype(self.dtype)
    # 
    Dataset.__init__(self, self.features.shape[0])


  def next_train_batch(self, batch_size):
    """Generate the next batch of samples
    Args:
      batch_size: <type 'int'> the number of samples in the batch
    Return:
      A tuple of features and labels of the samples in the batch
    """
    perm = self.generate_perm_for_train_batch(batch_size)
    return self.features[perm], self.labels[perm]


class NegDataset(Dataset):
  """Negative dataset class
  """
  def __init__(self, target_list, one_hot=True, dtype=np.float32):
    """Create a negative dataset for a protein kinase target.
      The data is read from a hdf5 file, pubchem_neg_sample.h5.
      Note that for each target, these samples has the corresponding labels,
      and I use a mask_dict to store these labels, i.e. mask_dict[target] = labels for target
    Args:
      target_list: <type 'list'> the protein kinase targets' list
      one_hot: <type 'bool'> whether to convert labels from dense to one_hot
      dtype: <type 'type'> data type of features 
    Return:
      None
    """
    # open h5 file
    self.h5_fn = os.path.join(h5_dir, "pubchem_neg_sample.h5")
    self.h5 = h5py.File(self.h5_fn, "r")
    # read ids
    self.ids = self.h5["chembl_id"].value
    # read 3 fp, and stack as feauture
    ap = sparse.csr_matrix((self.h5["ap"]["data"], self.h5["ap"]["indices"], self.h5["ap"]["indptr"]), shape=[len(self.h5["ap"]["indptr"]) - 1, vec_len])
    mg = sparse.csr_matrix((self.h5["mg"]["data"], self.h5["mg"]["indices"], self.h5["mg"]["indptr"]), shape=[len(self.h5["mg"]["indptr"]) - 1, vec_len])
    tt = sparse.csr_matrix((self.h5["tt"]["data"], self.h5["tt"]["indices"], self.h5["tt"]["indptr"]), shape=[len(self.h5["tt"]["indptr"]) - 1, vec_len])
    self.features = sparse.hstack([ap, mg, tt]).toarray()
    # label(mask)
    self.mask_dict = {}
    for target in target_list:
      #mask = self.h5["mask"][target].value
      mask = self.h5["cliped_mask"][target].value
      if one_hot == True:
        self.mask_dict[target] = dense_to_one_hot(mask)
      else:
        self.mask_dict[target] = mask
    # close h5 file
    self.h5.close()
    # dtype
    self.dtype = dtype
    # pre_process
    #self.features = np.log10(1.0 + self.features).astype(self.dtype)
    self.features = np.clip(self.features, 0, 1).astype(self.dtype)
    # 
    Dataset.__init__(self, self.features.shape[0])

  def next_train_batch(self, target, batch_size):
    """Generate the next batch of samples
    Args:
      batch_size: <type 'int'> the number of samples in the batch
    Return:
      A tuple of features and labels of the samples in the batch
    """
    perm = self.generate_perm_for_train_batch(batch_size)
    return self.features[perm], self.mask_dict[target][perm]


class Datasets(object):
  """dataset class, contains several positive datasets and one negative dataset.
  """
  def __init__(self, target_list, one_hot=True):
    """
    Args:
      target_list: <type 'list'> the protein kinase targets' list
      one_hot: <type 'bool'> whether to convert labels from dense to one_hot
    return:
      None
    """
    # read neg dataset
    self.neg = NegDataset(target_list, one_hot=one_hot)
    # read pos datasets
    self.pos = {}
    for target in target_list:
      self.pos[target] = PosDataset(target, one_hot=one_hot)

  def next_train_batch(self, target, pos_batch_size, neg_batch_size):
    """Generate the next batch of samples
    Args:
      target: <type 'str'> the positive target name
      pos_batch_size: <type 'int'> the number of samples in the batch from positive target dataset
      neg_batch_size: <type 'int'> the number of samples in the batch from negative target dataset  
    Return:
      A tuple of features and labels of the samples in the batch
    """
    pos_feature_batch, pos_label_batch = self.pos[target].next_train_batch(pos_batch_size)
    neg_feature_batch, neg_label_batch = self.neg.next_train_batch(target, neg_batch_size)
    return np.vstack([pos_feature_batch, neg_feature_batch]), np.vstack([pos_label_batch, neg_label_batch])
    
    
def test_dataset():
  """A simple test
  """
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
  d = Datasets(target_list)
  print("test for batching")
  print("batch_num     target feature_min feature_max label_min label_max")
  for step in range(2 * 500):
    for target in target_list:
      compds_batch, labels_batch = d.next_train_batch(target, 128, 128)
      if np.isnan(compds_batch).sum() > 0:
        print("warning: nan in feature"),
        print("%9d %10s %11.2f %11.2f %9.2f %9.2f" % (step, target, compds_batch.min(), compds_batch.max(), labels_batch.min(), labels_batch.max()))
      if (step % 500) == 0:
        print("%9d %10s %11.2f %11.2f %9.2f %9.2f" % (step, target, compds_batch.min(), compds_batch.max(), labels_batch.min(), labels_batch.max()))



if __name__ == "__main__":
  target_list = ["cdk2", "egfr_erbB1", "gsk3b", "hgfr", "map_k_p38a", "tpk_lck", "tpk_src", "vegfr2"]
  test_dataset()








