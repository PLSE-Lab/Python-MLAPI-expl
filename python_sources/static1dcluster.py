import numpy as np
import numba
import warnings
from sklearn.base import TransformerMixin, BaseEstimator


__all__ = ['Static1DCluster']

@numba.njit(parallel=True)
def _cluster(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(centroids, x)
    for i in numba.prange(len(indices)):
        if centroids[i] == x[i]:
            continue
        ind = indices[i]
        if ind >= len(centroids):
            indices[i] = len(centroids) - 1
        elif ind + 1 < len(centroids):
            dist = abs(centroids[ind] - x[i])
            dist2 = abs(centroids[ind + 1] - x[i])
            if dist > dist2:
                indices[i] = ind + 1
    return indices


class Static1DCluster(TransformerMixin, BaseEstimator):
    __slots__ = ['_centroids', 'dtype']
    
    def __init__(self, centroids=None, dtype=np.float32):
        super().__init__()
        self._centroids: np.ndarray = np.ndarray((0,), dtype=dtype)
        self.dtype = np.dtype(dtype)
        if centroids is not None:
            self._set_centroids(centroids)
            
    @property
    def shape(self):
        return self._centroids.shape
    
    def __len__(self):
        return self.shape[0]

    @property
    def centroids_(self) -> np.ndarray:
        return self._centroids
    
    @centroids_.setter
    def centoids_(self, value):
        self._set_centroids(value)

    def _set_centroids(self, data: np.ndarray):
        data = np.asanyarray(data, dtype=self.dtype)
        if data.ndim != 1:
            warnings.warn("centroids data has more than 1 dimension.")
            data = data.flatten()
        unique = np.unique(data) # np.unique sort the centoids
        if unique.shape[0] != data.shape[0]:
            warnings.warn("centroids data has duplicate values.")
        elif not np.array_equal(unique, data):
            warnings.warn("centroids data has been sorted.")
        self._centroids = unique

    def fit(self, data: np.ndarray):
        self._set_centroids(data)
        return self

    def transform(self, data: np.ndarray):
        indices = self.transform_indices(data)
        return self._centroids[indices]
    
    def transform_indices(self, data: np.ndarray):
        data = np.asanyarray(data, dtype=self.dtype)
        indices = _cluster(data, self._centroids)
        return indices

    def fit_transform(self, data: np.ndarray):
        return self.fit(data).transform(data)
    
    def __getstate__(self):
        return {'centroids_': self.centroids_, 'dtype': str(self.dtype)}
    
    def __setstate__(self, state):
        self.dtype = np.dtype(state['dtype'])
        self._set_centroids(state['centroids_'])
        
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    
    def __repr__(self):
        return "{cls}(centroids={centroids!s}, dtype={dtype!s})".format(cls=self.__class__.__name__,
                                                                    centroids=self.centroids_,
                                                                    dtype=np.dtype(self.dtype))
    
if __name__ == '__main__':
    c = Static1DCluster()
    c.fit([1, 2, 3])
    # transform() return closest cluster integer
    assert np.array_equal(c.transform([1, 2, 3, 4, 0, -1]), [1, 2, 3, 3, 1, 1])
    # transform_indices() return closest cluster integer's indice
    assert np.array_equal(c.transform_indices([1, 2, 3, 4, 0, -1]), [0, 1, 2, 2, 0, 0])
    
    c = Static1DCluster()
    c.fit([1, 3, 3, 0]) # given cluster points list contains duplicate & is unsorted
    assert np.array_equal(c.centroids_, [0, 1, 3])
    
    