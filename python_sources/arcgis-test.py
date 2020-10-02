#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ARG BASE_TAG=staging

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS nvidia
FROM gcr.io/kaggle-images/python-tensorflow-whl:2.1.0-py37-2 as tensorflow_whl
FROM gcr.io/kaggle-images/python:${BASE_TAG}

ADD clean-layer.sh  /tmp/clean-layer.sh

# Cuda support
COPY --from=nvidia /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg
# See b/142337634#comment28
RUN sed -i 's/deb https:\/\/developer.download.nvidia.com/deb http:\/\/developer.download.nvidia.com/' /etc/apt/sources.list.d/*.list

# Ensure the cuda libraries are compatible with the custom Tensorflow wheels.
# TODO(b/120050292): Use templating to keep in sync or COPY installed binaries from it.
ENV CUDA_MAJOR_VERSION=10
ENV CUDA_MINOR_VERSION=1
ENV CUDA_PATCH_VERSION=243
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION.$CUDA_PATCH_VERSION
ENV CUDA_PKG_VERSION=$CUDA_MAJOR_VERSION-$CUDA_MINOR_VERSION=$CUDA_VERSION-1
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# The stub is useful to us both for built-time linking and run-time linking, on CPU-only systems.
# When intended to be used with actual GPUs, make sure to (besides providing access to the host
# CUDA user libraries, either manually or through the use of nvidia-docker) exclude them. One
# convenient way to do so is to obscure its contents by a bind mount:
#   docker run .... -v /non-existing-directory:/usr/local/cuda/lib64/stubs:ro ...
ENV LD_LIBRARY_PATH_NO_STUBS="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION"
RUN apt-get update && apt-get install -y --no-install-recommends       cuda-cupti-$CUDA_PKG_VERSION       cuda-cudart-$CUDA_PKG_VERSION       cuda-cudart-dev-$CUDA_PKG_VERSION       cuda-libraries-$CUDA_PKG_VERSION       cuda-libraries-dev-$CUDA_PKG_VERSION       cuda-nvml-dev-$CUDA_PKG_VERSION       cuda-minimal-build-$CUDA_PKG_VERSION       cuda-command-line-tools-$CUDA_PKG_VERSION       libcudnn7=7.6.5.32-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION       libcudnn7-dev=7.6.5.32-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION       libnccl2=2.5.6-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION       libnccl-dev=2.5.6-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION &&     ln -s /usr/local/cuda-$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION /usr/local/cuda &&     ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 &&     /tmp/clean-layer.sh

# Install OpenCL & libboost (required by LightGBM GPU version)
RUN apt-get install -y ocl-icd-libopencl1 clinfo libboost-all-dev &&     mkdir -p /etc/OpenCL/vendors &&     echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd &&     /tmp/clean-layer.sh

# When using pip in a conda environment, conda commands should be ran first and then
# the remaining pip commands: https://www.anaconda.com/using-pip-in-a-conda-environment/
# However, because this image is based on the CPU image, this isn't possible but better
# to put them at the top of this file to minize conflicts.
RUN conda remove --force -y pytorch torchvision torchaudio cpuonly &&     conda install -y pytorch torchvision torchaudio cudatoolkit=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION -c pytorch &&     /tmp/clean-layer.sh

# Install LightGBM with GPU
RUN pip uninstall -y lightgbm &&     cd /usr/local/src &&     git clone --recursive https://github.com/microsoft/LightGBM &&     cd LightGBM &&     git checkout tags/v2.3.1 &&     mkdir build && cd build &&     cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. &&     make -j$(nproc) &&     cd /usr/local/src/LightGBM/python-package &&     python setup.py install --precompile &&     mkdir -p /etc/OpenCL/vendors &&     echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd &&     /tmp/clean-layer.sh

# Install JAX
# b/154150582#comment9: JAX 0.1.63 with jaxlib 0.1.43 is causing the GPU tests to hang.
ENV JAX_VERSION=0.1.62
ENV JAXLIB_VERSION=0.1.41
ENV JAX_PYTHON_VERSION=cp37
ENV JAX_CUDA_VERSION=cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION
ENV JAX_PLATFORM=linux_x86_64
ENV JAX_BASE_URL="https://storage.googleapis.com/jax-releases"

RUN  pip install $JAX_BASE_URL/$JAX_CUDA_VERSION/jaxlib-$JAXLIB_VERSION-$JAX_PYTHON_VERSION-none-$JAX_PLATFORM.whl &&      pip install jax==$JAX_VERSION

# Reinstall packages with a separate version for GPU support.
COPY --from=tensorflow_whl /tmp/tensorflow_gpu/*.whl /tmp/tensorflow_gpu/
RUN pip uninstall -y tensorflow &&     pip install /tmp/tensorflow_gpu/tensorflow*.whl &&     rm -rf /tmp/tensorflow_gpu &&     pip uninstall -y mxnet &&     # b/126259508 --no-deps prevents numpy from being downgraded.
    pip install --no-deps mxnet-cu$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION &&     /tmp/clean-layer.sh

# Install GPU-only packages
RUN pip install pycuda &&     pip install cupy-cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION &&     pip install pynvrtc &&     pip install nnabla-ext-cuda$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION &&     /tmp/clean-layer.sh

# Re-add TensorBoard Jupyter extension patch
# b/139212522 re-enable TensorBoard once solution for slowdown is implemented.
# ADD patches/tensorboard/notebook.py /opt/conda/lib/python3.7/site-packages/tensorboard/notebook.py

# Remove the CUDA stubs.
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NO_STUBS"


# In[ ]:


final Portal portal = new Portal("http://www.arcgis.com");
portal.addDoneLoadingListener(() -> {
  if (portal.getLoadStatus() == LoadStatus.LOADED) {
    PortalInfo portalInformation = portal.getPortalInfo();
    String portalName = portalInformation.getPortalName(); // Returns 'ArcGIS Online'
  }
});
portal.loadAsync();


# In[ ]:


Portal portal = new Portal("http://anorganization.maps.arcgis.com");


# In[ ]:


(/, Set, the, DefaultAuthenticationChallegeHandler, to, allow, authentication, with, the, portal.)
DefaultAuthenticationChallengeHandler handler = new DefaultAuthenticationChallengeHandler(activityContext);
AuthenticationManager.setAuthenticationChallengeHandler(handler);
(/, Create, a, Portal, object,, indicate, authentication, is, required)
final Portal portal = new Portal("http://www.arcgis.com", true);
portal.addDoneLoadingListener(new Runnable() {
  @Override
  public void run() {
    if (portal.getLoadStatus() == LoadStatus.LOADED) {
      PortalUser user = portal.getUser();
      String userDisplayName = user.getFullName(); // Returns display name of authenticated user
    }
  }
});
portal.loadAsync();


# In[ ]:


(/, Set, the, DefaultAuthenticationChallegeHandler, to, allow, authentication, with, the, portal.)
DefaultAuthenticationChallengeHandler handler = new DefaultAuthenticationChallengeHandler(this);
AuthenticationManager.setAuthenticationChallengeHandler(handler);
final Portal portal = new Portal("https://geoportal.mycompany.com", true);


# In[ ]:


final String serviceFeatureURL =
  "http://sampleserver5.arcgisonline.com/arcgis/rest/services/Elevation/WorldElevations/MapServer";
(/, create, new, ArcGISMap, image, Layer, from, service, url)
final ArcGISMapImageLayer imageLayer = new ArcGISMapImageLayer(serviceFeatureURL


# In[ ]:


(/, Create, a, DefaultAuthenticationChallengeHandler,, passing, in, an, Android, Context, (e.g., the, current, Activity))
DefaultAuthenticationChallengeHandler handler = new DefaultAuthenticationChallengeHandler(this);
(/, Set, the, challenge, handler, onto, the, AuthenticationManager)
AuthenticationManager.setAuthenticationChallengeHandler


# In[ ]:


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

