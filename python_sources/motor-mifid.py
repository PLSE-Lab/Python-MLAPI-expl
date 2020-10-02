# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import cv2
import warnings
from scipy import linalg
from tqdm import tqdm
from time import time


class MIFID():
    def __init__(self, model_path,
                 public_feature_path,
                 img_shape=(128, 128, 3),
                 output_shape=2048,
                 batch_size=32,
                 gpu_id=-1,
                 mem_fraction=0.2):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.img_size = img_shape[0]
        self.model = MotorbikeClassifier(model_path, gpu_id, mem_fraction, self.img_size)
        self.public_feature_path = public_feature_path
        self.FID_EPSILON = 10e-15

        self.COSINE_DISTANCE_THRESH = 0.05  # NO CHANGE DURING THE COMPETITION
        self.output_shape = output_shape
        self.VALID_NUM_IMAGES = 10000

        # public feature
        print('Load Public Embedding Features')
        with np.load(self.public_feature_path) as f:
            self.public_m2, self.public_s2, self.public_features2 = f['m'], f['s'], f['features']

    def get_public_feature(self):
        return self.public_m2, self.public_s2, self.public_features2

    def get_private_feature(self):
        return self.private_m2, self.private_s2, self.private_features2

    def preprocessing(self, np_arr):
        '''Preprocessing input of motorbike classifier'''
        np_arr = np_arr.astype(np.float)
        np_arr /= 255.0
        return np_arr

    def get_activations(self, images, batch_size=16, verbose=False):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
        -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                        must lie between 0 and 256.
        -- sess        : current session
        -- batch_size  : the images numpy array is split into batches with batch size
                        batch_size. A reasonable batch size depends on the disposable hardware.
        -- verbose    : If set to True and parameter out_step is given, the number of calculated
                        batches is reported.
        Returns:
        -- A numpy array of dimension (num images, 2048) that contains the
        activations of the given tensor when feeding inception with the query tensor.
        """

        n_images = images.shape[0]
        n_batches = np.ceil(n_images / batch_size).astype(int)
        feature_arr = np.empty((n_images, self.output_shape))
        print('------------------------------------------------------')
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            if start + batch_size < n_images:
                end = start + batch_size
            else:
                end = n_images
            batch = images[start:end]
            batch = self.preprocessing(batch)
            _, features = self.model.predict(batch)
            feature_arr[start:end] = features
        print("done")
        print('------------------------------------------------------')
        return feature_arr

    def normalize_rows(self, x: np.ndarray):
        """
        function that normalizes each row of the matrix x to have unit length.

        Args:
        ``x``: A numpy matrix of shape (n, m)

        Returns:
        ``x``: The normalized (by row) numpy matrix.
        """
        return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

    def cosine_distance(self, features1, features2):
        print('rows of zeros in features1 = {}'.format(sum(np.sum(features1, axis=1) == 0)))
        print('rows of zeros in features2 = {}'.format(sum(np.sum(features2, axis=1) == 0)))
        features1_nozero = features1[np.sum(features1, axis=1) != 0]
        features2_nozero = features2[np.sum(features2, axis=1) != 0]
        norm_f1 = self.normalize_rows(features1_nozero)
        norm_f2 = self.normalize_rows(features2_nozero)

        d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))
        print('d.shape= {}'.format(d.shape))
        print('np.min(d, axis=1).shape={}'.format(np.min(d, axis=1).shape))
        mean_min_d = np.mean(np.min(d, axis=1))
        print('distance={}'.format(mean_min_d))
        return mean_min_d

    def distance_thresholding(self, d, eps):
        if d < eps:
            return d
        else:
            return 1

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                inception net ( like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                precalcualted on an representive data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # product might be almost singular
        t = time()
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        print('- Compute sqrtm in {}'.format(time() - t))

        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            print('[WARNING] {}'.format(msg))
            print('Recompute sqrtm ...')
            offset = np.eye(sigma1.shape[0]) * eps
            # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            t = time()
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            print('- Compute sqrtm in {}'.format(time() - t))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(
                    "Imaginary component {}, quantity of input images must be larger than 2048".format(m))
            covmean = covmean.real

        # covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1,sigma2))
        print('covmean.shape={}'.format(covmean.shape))
        # tr_covmean = tf.linalg.trace(covmean)

        t = time()
        tr_covmean = np.trace(covmean)
        print('- Compute trace in {}'.format(time() - t))
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        # return diff.dot(diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean

    # -------------------------------------------------------------------------------

    def calculate_activation_statistics(self, np_imgs, batch_size=32, verbose=False):
        """Calculation of the statistics used by the FID.
        Params:
        -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                        must lie between 0 and 255.
        -- sess        : current session
        -- batch_size  : the images numpy array is split into batches with batch size
                        batch_size. A reasonable batch size depends on the available hardware.
        -- verbose     : If set to True and parameter out_step is given, the number of calculated
                        batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the incption model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the incption model.
        """
        act = self.get_activations(np_imgs, batch_size, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma, act

    def _handle_path_memorization(self, np_imgs):
        print('Compute embedding features ...')
        m, s, features = self.calculate_activation_statistics(np_imgs, verbose=True)
        del np_imgs  # clean up memory
        return m, s, features

    def calculate_kid_given_paths(self, np_imgs):
        ''' Calculates the KID of two paths. '''
        # from user
        m1, s1, features1 = self._handle_path_memorization(np_imgs)

        # public feature
        m2, s2, features2 = self.get_public_feature()

        print('m1, m2 shape = {}, {}'.format(m1.shape, m2.shape))
        print('s1, s2 shape = {}, {}'.format(s1.shape, s2.shape))

        print('\nStarting calculating FID')
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        print('Done with FID, starting distance calculation')
        distance = self.cosine_distance(features1, features2)
        return fid_value, distance

    def list2numpy(self, img_arr_list):
        np_imgs = np.zeros((len(img_arr_list), self.img_size, self.img_size, 3), dtype=np.uint8)
        for idx, img in enumerate(img_arr_list):
            np_imgs[idx] = img
        return np_imgs

    def process_list_images(self, imgArrList):
        assert len(imgArrList) == self.VALID_NUM_IMAGES, ValueError(
            'Require %s images to evaluate' % self.VALID_NUM_IMAGES)
        np_imgs = self.list2numpy(imgArrList)
        score = self.compute_mifid(np_imgs)
        return score

    def compute_mifid(self, np_imgs):
        fid_value, distance = self.calculate_kid_given_paths(np_imgs)
        print('Compute distance with threshold={}\n'.format(self.COSINE_DISTANCE_THRESH))
        distance = self.distance_thresholding(distance, self.COSINE_DISTANCE_THRESH)
        print("FID: {}".format(fid_value))
        print("distance: {}".format(distance))
        print("Final Score: {}".format(fid_value / (distance + self.FID_EPSILON)))
        return fid_value / (distance + self.FID_EPSILON)


class MotorbikeClassifier():
    def __init__(self, model_path, gpu_id=-1, mem_fraction=0.2, img_size=128):
        # self.tf, self.sess, self.graph = get_tf_env(gpu_id=gpu_id, mem_fraction=mem_fraction)
        self.tf, self.config = get_tf_env(gpu_id=gpu_id, mem_fraction=mem_fraction)
        self.img_size = img_size
        self.PATH_TO_MODEL = model_path

        print('[Motorbike Classifer] Load Motorbike Classifer from {}'.format(self.PATH_TO_MODEL))
        self.graph = self.tf.Graph()
        self.sess = self.tf.Session(graph=self.graph, config=self.config)

        with self.graph.as_default():
            od_graph_def = self.tf.GraphDef()
            with self.tf.gfile.GFile(self.PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                self.tf.import_graph_def(od_graph_def, name='')
            self.input_tensor = self.graph.get_tensor_by_name('input_1:0')
            self.output_tensor = self.graph.get_tensor_by_name('activation_95/Sigmoid:0')
            self.embedding_tensor = self.graph.get_tensor_by_name('global_average_pooling2d_1/Mean:0')

    def predict(self, img_expanded):
        try:
            with self.graph.as_default():
                scores, embs = self.sess.run([self.output_tensor, self.embedding_tensor],
                                             feed_dict={self.input_tensor: img_expanded})
            return scores, embs
        except:
            import traceback
            print(traceback.print_exc())


def get_tf_env(gpu_id, mem_fraction):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = float(mem_fraction)
    return tf, config