#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import logging
import re
import subprocess
 
 
def pip_show(package_name, packages=[]):
    if package_name in packages:
        return # avoid checking the same package twice if multiple packages depends on it.
    packages.append(package_name)
 
    result = subprocess.run(['pip', 'show', package_name], stdout=subprocess.PIPE)
    if result.returncode != 0:
        logging.error("pip show %s failed", package_name)
    
    show_stdout = result.stdout.decode("utf-8")
    print(package_name + "==" + get_version(show_stdout))
 
    for dependency in get_dependencies(show_stdout):
        pip_show(dependency, packages=packages)
 
def get_version(show_stdout):
    for line in show_stdout.split("\n"):
        m = re.match(r"^Version:\s(?P<version>.+)$", line)
        if m:
            return m.group('version')
    return "not found"
 
def get_dependencies(show_stdout):
    for line in show_stdout.split("\n"):
        m = re.match(r"^Requires:\s(?P<requires>.+)$", line)
        if m:
            return m.group('requires').split(', ')
    return []
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=str, help='package name')
    args = parser.parse_args()
    
    pip_show(args.package)
 


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


FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 AS nvidia
FROM gcr.io/deeplearning-platform-release/base-cpu:latest
 
# Avoid interactive configuration prompts/dialogs during apt-get.
ENV DEBIAN_FRONTEND=noninteractive
 
# This is necessary to for apt to access HTTPS sources
RUN apt-get update &&     apt-get install apt-transport-https
# Cuda support
COPY --from=nvidia /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/
COPY --from=nvidia /etc/apt/trusted.gpg /etc/apt/trusted.gpg.d/cuda.gpg
# See b/142337634#comment28
RUN sed -i 's/deb https:\/\/developer.download.nvidia.com/deb http:\/\/developer.download.nvidia.com/' /etc/apt/sources.list.d/*.list
# Ensure the cuda libraries are compatible with the GPU image.
# TODO(b/120050292): Use templating to keep in sync.
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
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION"
RUN apt-get update && apt-get install -y --no-install-recommends       cuda-cupti-$CUDA_PKG_VERSION       cuda-cudart-$CUDA_PKG_VERSION       cuda-cudart-dev-$CUDA_PKG_VERSION       cuda-libraries-$CUDA_PKG_VERSION       cuda-libraries-dev-$CUDA_PKG_VERSION       cuda-nvml-dev-$CUDA_PKG_VERSION       cuda-minimal-build-$CUDA_PKG_VERSION       cuda-command-line-tools-$CUDA_PKG_VERSION       libcudnn7=7.6.5.32-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION       libcudnn7-dev=7.6.5.32-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION       libnccl2=2.5.6-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION       libnccl-dev=2.5.6-1+cuda$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION &&     ln -s /usr/local/cuda-$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION /usr/local/cuda &&     ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN pip install --upgrade pip
# See _TF_(MIN|MAX)_BAZEL_VERSION at https://github.com/tensorflow/tensorflow/blob/master/configure.py.
ENV BAZEL_VERSION=0.29.1
RUN apt-get install -y gnupg zip openjdk-8-jdk &&     apt-get install -y --no-install-recommends       bash-completion       zlib1g-dev &&     wget --no-verbose "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb" &&     dpkg -i bazel_*.deb &&     rm bazel_*.deb
# Fetch tensorflow & install dependencies.
RUN cd /usr/local/src &&     git clone https://github.com/tensorflow/tensorflow &&     cd tensorflow &&     git checkout tags/v2.1.0 &&     pip install keras_applications --no-deps &&     pip install keras_preprocessing --no-deps
# Create a tensorflow wheel for CPU
RUN cd /usr/local/src/tensorflow &&     cat /dev/null | ./configure &&     bazel build --config=opt                 --config=v2                 --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"                 //tensorflow/tools/pip_package:build_pip_package &&     bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_cpu &&     bazel clean
# Create a tensorflow wheel for GPU/cuda
ENV TF_NEED_CUDA=1
ENV TF_CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION
# 3.7 is for the K80 and 6.0 is for the P100, 7.5 is for the T4: https://developer.nvidia.com/cuda-gpus
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.7,6.0,7.5
ENV TF_CUDNN_VERSION=7
ENV TF_NCCL_VERSION=2
ENV NCCL_INSTALL_PATH=/usr/
 
RUN cd /usr/local/src/tensorflow &&     # TF_NCCL_INSTALL_PATH is used for both libnccl.so.2 and libnccl.h. Make sure they are both accessible from the same directory.
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/ &&     cat /dev/null | ./configure &&     echo "/usr/local/cuda-${TF_CUDA_VERSION}/targets/x86_64-linux/lib/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig &&     bazel build --config=opt                 --config=v2                 --config=cuda                 --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"                 //tensorflow/tools/pip_package:build_pip_package &&     rm /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig &&     bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_gpu &&     bazel clean
 
ADD tensorflow-gcs-config /usr/local/src/tensorflow_gcs_config/
# Build tensorflow_gcs_config library against the tensorflow_cpu build
RUN cd /usr/local/src/tensorflow_gcs_config &&     apt-get install -y libcurl4-openssl-dev &&     pip install /tmp/tensorflow_cpu/tensorflow*.whl &&     python setup.py bdist_wheel -d /tmp/tensorflow_gcs_config &&     bazel clean
# Print out the built .whl files
RUN ls -R /tmp/tensorflow*


# In[ ]:


"""TensorFlow Data Validation external dependencies that can be loaded in WORKSPACE files.
"""
 
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
 
def tf_data_validation_workspace():
    """All TensorFlow Data Validation external dependencies."""
    tf_workspace(
        path_prefix = "",
        tf_repo_name = "org_tensorflow",
    )
 
    # LINT.IfChange
    # The next line (a comment) is important because it is used to
    # locate the git_repository repo rule. Therefore if it's changed, also
    # change copy.bara.sky.
    #
    # Fetch tf.Metadata repo from GitHub.
    git_repository(
        name = "com_github_tensorflow_metadata",
        commit = "7225352f33a07367a205ca8132e85272e034b7a8",
        remote = "https://github.com/tensorflow/metadata.git",
    )
    # LINT.ThenChange(//third_party/py/tensorflow_data_validation/google/copy.bara.sky)
 


# In[ ]:


*(Copyright, 2018, Google, LLC)
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    https://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
 
#include "tensorflow_data_validation/anomalies/schema.h"
 
#include <map>
#include <memory>
#include <set>
#include <string>
 
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/bool_domain_util.h"
#include "tensorflow_data_validation/anomalies/custom_domain_util.h"
#include "tensorflow_data_validation/anomalies/dataset_constraints_util.h"
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/float_domain_util.h"
#include "tensorflow_data_validation/anomalies/int_domain_util.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_data_validation/anomalies/string_domain_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"
 
namespace tensorflow {
namespace data_validation {
namespace {
using ::absl::optional;
using ::tensorflow::Status;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::SparseFeature;
using ::tensorflow::metadata::v0::StringDomain;
using ::tensorflow::metadata::v0::WeightedFeature;
using PathProto = ::tensorflow::metadata::v0::Path;
 
constexpr char kTrainingServingSkew[] = "Training/Serving skew";
 
// LINT.IfChange(sparse_feature_custom_stat_names)
static constexpr char kMissingSparseValue[] = "missing_value";
static constexpr char kMissingSparseIndex[] = "missing_index";
static constexpr char kMaxLengthDiff[] = "max_length_diff";
static constexpr char kMinLengthDiff[] = "min_length_diff";
// LINT.ThenChange(../statistics/generators/sparse_feature_stats_generator.py:custom_stat_names)
 
// LINT.IfChange(weighted_feature_custom_stat_names)
static constexpr char kMissingWeightedValue[] = "missing_value";
static constexpr char kMissingWeight[] = "missing_weight";
static constexpr char kMaxWeightLengthDiff[] = "max_weight_length_diff";
static constexpr char kMinWeightLengthDiff[] = "min_weight_length_diff";
// LINT.ThenChange(../statistics/generators/weighted_feature_stats_generator.py:custom_stat_names)
 
template <typename Container>
bool ContainsValue(const Container& a, const string& value) {
  return absl::c_find(a, value) != a.end();
}
 
std::set<tensorflow::metadata::v0::FeatureType> AllowedFeatureTypes(
    Feature::DomainInfoCase domain_info_case) {
  switch (domain_info_case) {
    case Feature::kDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kBoolDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES,
              tensorflow::metadata::v0::FLOAT};
    case Feature::kIntDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::kFloatDomain:
      return {tensorflow::metadata::v0::FLOAT, tensorflow::metadata::v0::BYTES};
    case Feature::kStringDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kStructDomain:
      return {tensorflow::metadata::v0::STRUCT};
    case Feature::kNaturalLanguageDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kImageDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kMidDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kUrlDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kTimeDomain:
      // Consider also supporting time as floats.
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::DOMAIN_INFO_NOT_SET:
      ABSL_FALLTHROUGH_INTENDED;
    default:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::FLOAT,
              tensorflow::metadata::v0::BYTES,
              tensorflow::metadata::v0::STRUCT};
  }
}
 
// Remove all elements from the input array for which the input predicate
// pred is true. Returns number of erased elements.
template <typename T, typename Predicate>
int RemoveIf(::tensorflow::protobuf::RepeatedPtrField<T>* array,
             const Predicate& pred) {
  int i = 0, end = array->size();
  while (i < end && !pred(&array->Get(i))) ++i;
 
  if (i == end) return 0;
 
  (/, 'i', is, positioned, at, first, element, to, be, removed.)
  for (int j = i + 1; j < end; ++j) {
    if (!pred(&array->Get(j))) array->SwapElements(j, i++);
  }
 
  array->DeleteSubrange(i, end - i);
  return end - i;
}
 
Feature* GetExistingFeatureHelper(
    const string& last_part,
    tensorflow::protobuf::RepeatedPtrField<Feature>* features) {
  for (tensorflow::metadata::v0::Feature& feature : *features) {
    if (feature.name() == last_part) {
      return &feature;
    }
  }
  return nullptr;
}
 
void ClearStringDomainHelper(
    const string& domain_name,
    tensorflow::protobuf::RepeatedPtrField<Feature>* features) {
  for (tensorflow::metadata::v0::Feature& feature : *features) {
    if (feature.domain() == domain_name) {
      ::tensorflow::data_validation::ClearDomain(&feature);
    }
    if (feature.has_struct_domain()) {
      ClearStringDomainHelper(
          domain_name, feature.mutable_struct_domain()->mutable_feature());
    }
  }
}
 
SparseFeature* GetExistingSparseFeatureHelper(
    const string& name,
    tensorflow::protobuf::RepeatedPtrField<
        tensorflow::metadata::v0::SparseFeature>* sparse_features) {
  for (SparseFeature& sparse_feature : *sparse_features) {
    if (sparse_feature.name() == name) {
      return &sparse_feature;
    }
  }
  return nullptr;
}
 
(/, absl::nullopt, is, the, set, of, all, paths.)
bool ContainsPath(const absl::optional<std::set<Path>>& paths_to_consider,
                  const Path& path) {
  if (!paths_to_consider) {
    return true;
  }
  return ContainsKey(*paths_to_consider, path);
}
 
}  // namespace
 
Status Schema::Init(const tensorflow::metadata::v0::Schema& input) {
  if (!IsEmpty()) {
    return InvalidArgument("Schema is not empty when Init() called.");
  }
  schema_ = input;
  return Status::OK();
}
 
Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config) {
  return Update(dataset_stats, Updater(config), absl::nullopt);
}
 
Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config,
                      const std::vector<Path>& paths_to_consider) {
  return Update(
      dataset_stats, Updater(config),
      std::set<Path>(paths_to_consider.begin(), paths_to_consider.end()));
}
 
tensorflow::Status Schema::UpdateFeature(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    std::vector<Description>* descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;
 
  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  SparseFeature* sparse_feature =
      GetExistingSparseFeature(feature_stats_view.GetPath());
  WeightedFeature* weighted_feature =
      GetExistingWeightedFeature(feature_stats_view.GetPath());
  if (weighted_feature != nullptr) {
    if ((feature != nullptr || sparse_feature != nullptr) &&
        !::tensorflow::data_validation::WeightedFeatureIsDeprecated(
            *weighted_feature)) {
      descriptions->push_back({tensorflow::metadata::v0::AnomalyInfo::
                                   WEIGHTED_FEATURE_NAME_COLLISION,
                               "Weighted feature name collision",
                               "Weighted feature name collision."});
      ::tensorflow::data_validation::DeprecateWeightedFeature(weighted_feature);
      if (feature != nullptr) {
        ::tensorflow::data_validation::DeprecateFeature(feature);
      }
      if (sparse_feature != nullptr) {
        ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      }
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    } else {
      *descriptions =
          UpdateWeightedFeature(feature_stats_view, weighted_feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    }
  }
 
  if (sparse_feature != nullptr &&
      !::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature)) {
    if (feature != nullptr &&
        !::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
      descriptions->push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_NAME_COLLISION,
           "Sparse feature name collision", "Sparse feature name collision."});
      ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      ::tensorflow::data_validation::DeprecateFeature(feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    } else {
      *descriptions = UpdateSparseFeature(feature_stats_view, sparse_feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    }
  }
 
  if (feature != nullptr) {
    *descriptions = UpdateFeatureInternal(updater, feature_stats_view, feature);
    updater.UpdateSeverityForAnomaly(*descriptions, severity);
    return Status::OK();
  } else {
    const Description description = {
        tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN, "New column",
        "New column (column in data but not in schema)"};
    *descriptions = {description};
    updater.UpdateSeverityForAnomaly(*descriptions, severity);
    return updater.CreateColumn(feature_stats_view, this, severity);
  }
  return Status::OK();
}
 
bool Schema::FeatureIsDeprecated(const Path& path) {
  Feature* feature = GetExistingFeature(path);
  if (feature == nullptr) {
    SparseFeature* sparse_feature = GetExistingSparseFeature(path);
    if (sparse_feature != nullptr) {
      return ::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature);
    }
    // Here, the result is undefined.
    return false;
  }
  return ::tensorflow::data_validation::FeatureIsDeprecated(*feature);
}
 
void Schema::DeprecateFeature(const Path& path) {
  ::tensorflow::data_validation::DeprecateFeature(
      CHECK_NOTNULL(GetExistingFeature(path)));
}
 
Status Schema::UpdateRecursively(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    const absl::optional<std::set<Path>>& paths_to_consider,
    std::vector<Description>* descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;
  if (!ContainsPath(paths_to_consider, feature_stats_view.GetPath())) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFeature(updater, feature_stats_view, descriptions, severity));
  if (!FeatureIsDeprecated(feature_stats_view.GetPath())) {
    for (const FeatureStatsView& child : feature_stats_view.GetChildren()) {
      std::vector<Description> child_descriptions;
      tensorflow::metadata::v0::AnomalyInfo::Severity child_severity;
      TF_RETURN_IF_ERROR(UpdateRecursively(updater, child, paths_to_consider,
                                           &child_descriptions,
                                           &child_severity));
      descriptions->insert(descriptions->end(), child_descriptions.begin(),
                           child_descriptions.end());
      *severity = MaxSeverity(child_severity, *severity);
    }
  }
  updater.UpdateSeverityForAnomaly(*descriptions, severity);
  return Status::OK();
}
 
Schema::Updater::Updater(const FeatureStatisticsToProtoConfig& config)
    : config_(config),
      columns_to_ignore_(config.column_to_ignore().begin(),
                         config.column_to_ignore().end()) {
  for (const ColumnConstraint& constraint : config.column_constraint()) {
    for (const PathProto& column_path : constraint.column_path()) {
      grouped_enums_[Path(column_path)] = constraint.enum_name();
    }
  }
}
 
(/, Sets, the, severity, based, on, anomaly, descriptions,, possibly, using, severity)
(/, overrides.)
void Schema::Updater::UpdateSeverityForAnomaly(
    const std::vector<Description>& descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  for (const auto& description : descriptions) {
    // By default, all anomalies are ERROR level.
    tensorflow::metadata::v0::AnomalyInfo::Severity severity_for_anomaly =
        tensorflow::metadata::v0::AnomalyInfo::ERROR;
 
    if (config_.new_features_are_warnings() &&
        (description.type ==
         tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN)) {
      LOG(WARNING) << "new_features_are_warnings is deprecated. Use "
                      "severity_overrides";
      severity_for_anomaly = tensorflow::metadata::v0::AnomalyInfo::WARNING;
    }
    for (const auto& severity_override : config_.severity_overrides()) {
      if (severity_override.type() == description.type) {
        severity_for_anomaly = severity_override.severity();
      }
    }
    *severity = MaxSeverity(*severity, severity_for_anomaly);
  }
}
 
Status Schema::Updater::CreateColumn(
    const FeatureStatsView& feature_stats_view, Schema* schema,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  if (schema->GetExistingFeature(feature_stats_view.GetPath()) != nullptr) {
    return InvalidArgument("Schema already contains \"",
                           feature_stats_view.GetPath().Serialize(), "\".");
  }
 
  Feature* feature = schema->GetNewFeature(feature_stats_view.GetPath());
 
  feature->set_type(feature_stats_view.GetFeatureType());
  InitValueCountAndPresence(feature_stats_view, feature);
  if (ContainsKey(columns_to_ignore_,
                  feature_stats_view.GetPath().Serialize())) {
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return Status::OK();
  }
 
  if (BestEffortUpdateCustomDomain(feature_stats_view.custom_stats(),
                                   feature)) {
    return Status::OK();
  } else if (ContainsKey(grouped_enums_, feature_stats_view.GetPath())) {
    const string& enum_name = grouped_enums_.at(feature_stats_view.GetPath());
    StringDomain* result = schema->GetExistingStringDomain(enum_name);
    if (result == nullptr) {
      result = schema->GetNewStringDomain(enum_name);
    }
    UpdateStringDomain(*this, feature_stats_view, 0, result);
    return Status::OK();
  } else if (feature_stats_view.HasInvalidUTF8Strings() ||
             feature_stats_view.type() == FeatureNameStatistics::BYTES) {
    // If there are invalid UTF8 strings, or the field should not be further
    // interpreted, add no domain info.
    return Status::OK();
  } else if (IsBoolDomainCandidate(feature_stats_view)) {
    *feature->mutable_bool_domain() = BoolDomainFromStats(feature_stats_view);
    return Status::OK();
  } else if (IsIntDomainCandidate(feature_stats_view)) {
    // By default don't set any values.
    feature->mutable_int_domain();
    return Status::OK();
  } else if (IsStringDomainCandidate(feature_stats_view,
                                     config_.enum_threshold())) {
    StringDomain* string_domain =
        schema->GetNewStringDomain(feature_stats_view.GetPath().Serialize());
    UpdateStringDomain(*this, feature_stats_view, 0, string_domain);
    *feature->mutable_domain() = string_domain->name();
    return Status::OK();
  } else {
    (/, No, domain, info, for, this, field.)
    return Status::OK();
  }
}
 
(/, Returns, true, if, there, is, a, limit, on, the, size, of, a, string, domain, and, it)
(/, should, be, deleted.)
bool Schema::Updater::string_domain_too_big(int size) const {
  return config_.has_enum_delete_threshold() &&
         config_.enum_delete_threshold() <= size;
}
 
bool Schema::IsEmpty() const {
  return schema_.feature().empty() && schema_.string_domain().empty();
}
 
void Schema::Clear() { schema_.Clear(); }
 
StringDomain* Schema::GetNewStringDomain(const string& candidate_name) {
  std::set<string> names;
  for (const StringDomain& string_domain : schema_.string_domain()) {
    names.insert(string_domain.name());
  }
  string new_name = candidate_name;
  int index = 1;
  while (ContainsKey(names, new_name)) {
    ++index;
    new_name = absl::StrCat(candidate_name, index);
  }
  StringDomain* result = schema_.add_string_domain();
  *result->mutable_name() = new_name;
  return result;
}
 
StringDomain* Schema::GetExistingStringDomain(const string& name) {
  for (int i = 0; i < schema_.string_domain_size(); ++i) {
    StringDomain* possible = schema_.mutable_string_domain(i);
    if (possible->name() == name) {
      return possible;
    }
  }
 
  // If there is no match, return nullptr.
  return nullptr;
}
 
std::vector<std::set<string>> Schema::SimilarEnumTypes(
    const EnumsSimilarConfig& config) const {
  std::vector<bool> used(schema_.string_domain_size(), false);
  std::vector<std::set<string>> result;
  for (int index_a = 0; index_a < schema_.string_domain_size(); ++index_a) {
    if (!used[index_a]) {
      const StringDomain& string_domain_a = schema_.string_domain(index_a);
      std::set<string> similar;
      for (int index_b = index_a + 1; index_b < schema_.string_domain_size();
           ++index_b) {
        if (!used[index_b]) {
          const StringDomain& string_domain_b = schema_.string_domain(index_b);
          if (IsSimilarStringDomain(string_domain_a, string_domain_b, config)) {
            similar.insert(string_domain_b.name());
          }
        }
      }
      if (!similar.empty()) {
        similar.insert(string_domain_a.name());
        result.push_back(similar);
      }
    }
  }
  return result;
}
 
std::vector<Path> Schema::GetAllRequiredFeatures(
    const Path& prefix,
    const tensorflow::protobuf::RepeatedPtrField<Feature>& features,
    const absl::optional<string>& environment) const {
  // This recursively walks through the structure. Sometimes, a feature is
  // not required because its parent is deprecated.
  std::vector<Path> result;
  for (const Feature& feature : features) {
    const Path child_path = prefix.GetChild(feature.name());
    if (IsExistenceRequired(feature, environment)) {
      result.push_back(child_path);
    }
    // There is an odd semantics here. Here, if a child feature is required,
    // but the parent is not, we could have an anomaly for the missing child
    // feature, even though it is the parent that is actually missing.
    if (!::tensorflow::data_validation::FeatureIsDeprecated(feature)) {
      std::vector<Path> descendants = GetAllRequiredFeatures(
          child_path, feature.struct_domain().feature(), environment);
      result.insert(result.end(), descendants.begin(), descendants.end());
    }
  }
  return result;
}
 
std::vector<Path> Schema::GetMissingPaths(
    const DatasetStatsView& dataset_stats) {
  std::set<Path> paths_present;
  for (const FeatureStatsView& feature_stats_view : dataset_stats.features()) {
    paths_present.insert(feature_stats_view.GetPath());
  }
  std::vector<Path> paths_absent;
 
  for (const Path& path : GetAllRequiredFeatures(Path(), schema_.feature(),
                                                 dataset_stats.environment())) {
    if (!ContainsKey(paths_present, path)) {
      paths_absent.push_back(path);
    }
  }
  return paths_absent;
}
 
(/, TODO(b/148406484):, currently,, only, looks, at, top-level, features.)
(/, Make, this, include, lower, level, features, as, well.)
(/, See, also, b/114757721.)
std::map<string, std::set<Path>> Schema::EnumNameToPaths() const {
  std::map<string, std::set<Path>> result;
  for (const Feature& feature : schema_.feature()) {
    if (feature.has_domain()) {
      result[feature.domain()].insert(Path({feature.name()}));
    }
  }
  return result;
}
 
Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const Updater& updater,
                      const absl::optional<std::set<Path>>& paths_to_consider) {
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity severity;
 
  for (const auto& feature_stats_view : dataset_stats.GetRootFeatures()) {
    TF_RETURN_IF_ERROR(UpdateRecursively(updater, feature_stats_view,
                                         paths_to_consider, &descriptions,
                                         &severity));
  }
  for (const Path& missing_path : GetMissingPaths(dataset_stats)) {
    if (ContainsPath(paths_to_consider, missing_path)) {
      DeprecateFeature(missing_path);
    }
  }
  return Status::OK();
}
 
(/, TODO(b/114757721):, expose, this.)
Status Schema::GetRelatedEnums(const DatasetStatsView& dataset_stats,
                               FeatureStatisticsToProtoConfig* config) {
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Update(dataset_stats, *config));
 
  std::vector<std::set<string>> similar_enums =
      schema.SimilarEnumTypes(config->enums_similar_config());
  // Map the enum names to the paths.
  const std::map<string, std::set<Path>> enum_name_to_paths =
      schema.EnumNameToPaths();
  for (const std::set<string>& set : similar_enums) {
    if (set.empty()) {
      return Internal("Schema::SimilarEnumTypes returned an empty set.");
    }
    ColumnConstraint* column_constraint = config->add_column_constraint();
    for (const string& enum_name : set) {
      if (ContainsKey(enum_name_to_paths, enum_name)) {
        for (const auto& column : enum_name_to_paths.at(enum_name)) {
          *column_constraint->add_column_path() = column.AsProto();
        }
      }
    }
    // Choose the shortest name for the enum.
    string best_name = *set.begin();
    for (const string& current_name : set) {
      if (current_name.size() < best_name.size()) {
        best_name = current_name;
      }
    }
    *column_constraint->mutable_enum_name() = best_name;
  }
  return Status::OK();
}
 
tensorflow::metadata::v0::Schema Schema::GetSchema() const { return schema_; }
 
bool Schema::FeatureExists(const Path& path) {
  return GetExistingFeature(path) != nullptr ||
         GetExistingSparseFeature(path) != nullptr ||
         GetExistingWeightedFeature(path) != nullptr;
}
 
Feature* Schema::GetExistingFeature(const Path& path) {
  if (path.size() == 1) {
    return GetExistingFeatureHelper(path.last_step(),
                                    schema_.mutable_feature());
  } else {
    Path parent = path.GetParent();
    Feature* parent_feature = GetExistingFeature(parent);
    if (parent_feature == nullptr) {
      return nullptr;
    }
    if (!parent_feature->has_struct_domain()) {
      return nullptr;
    }
    return GetExistingFeatureHelper(
        path.last_step(),
        parent_feature->mutable_struct_domain()->mutable_feature());
  }
  return nullptr;
}
 
SparseFeature* Schema::GetExistingSparseFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() == 1) {
    return GetExistingSparseFeatureHelper(path.last_step(),
                                          schema_.mutable_sparse_feature());
  } else {
    Feature* parent_feature = GetExistingFeature(path.GetParent());
    if (parent_feature == nullptr) {
      return nullptr;
    }
    if (!parent_feature->has_struct_domain()) {
      return nullptr;
    }
    return GetExistingSparseFeatureHelper(
        path.last_step(),
        parent_feature->mutable_struct_domain()->mutable_sparse_feature());
  }
}
 
WeightedFeature* Schema::GetExistingWeightedFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() != 1) {
    // Weighted features are always top-level features with single-step paths.
    return nullptr;
  }
  auto name = path.last_step();
  for (WeightedFeature& weighted_feature :
       *schema_.mutable_weighted_feature()) {
    if (weighted_feature.name() == name) {
      return &weighted_feature;
    }
  }
  return nullptr;
}
 
Feature* Schema::GetNewFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() > 1) {
    Path parent = path.GetParent();
    Feature* parent_feature = CHECK_NOTNULL(GetExistingFeature(parent));
    Feature* result = parent_feature->mutable_struct_domain()->add_feature();
    *result->mutable_name() = path.last_step();
    return result;
  } else {
    Feature* result = schema_.add_feature();
    *result->mutable_name() = path.last_step();
    return result;
  }
}
 
::tensorflow::metadata::v0::DatasetConstraints*
Schema::GetExistingDatasetConstraints() {
  if (schema_.has_dataset_constraints()) {
    return schema_.mutable_dataset_constraints();
  }
  return nullptr;
}
 
bool Schema::IsFeatureInEnvironment(
    const Feature& feature, const absl::optional<string>& environment) const {
  if (environment) {
    if (ContainsValue(feature.in_environment(), *environment)) {
      return true;
    }
    if (ContainsValue(feature.not_in_environment(), *environment)) {
      return false;
    }
    if (ContainsValue(schema_.default_environment(), *environment)) {
      return true;
    }
    return false;
  }
  // If environment is not set, then the feature is considered in the
  // environment by default.
  return true;
}
 
bool Schema::IsExistenceRequired(
    const Feature& feature, const absl::optional<string>& environment) const {
  if (::tensorflow::data_validation::FeatureIsDeprecated(feature)) {
    return false;
  }
  if (feature.presence().min_count() <= 0 &&
      feature.presence().min_fraction() <= 0.0) {
    return false;
  }
  // If a feature is in the environment, it is required.
  return IsFeatureInEnvironment(feature, environment);
}
 
(/, TODO(b/148406400):, Switch, AnomalyInfo::Type, from, UNKNOWN_TYPE.)
(/, TODO(b/148406994):, Handle, missing, FeatureType, more, elegantly,, inferring, it)
(/, when, necessary.)
std::vector<Description> Schema::UpdateFeatureSelf(Feature* feature) {
  std::vector<Description> descriptions;
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }
  if (!feature->has_name()) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat(
             "unspecified name (maybe meant to be the empty string): find "
             "name rather than deprecating.")});
    (/, Deprecating, the, feature, is, the, only, possible, "fix", here.)
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return descriptions;
  }
 
  if (!feature->has_type()) {
    (/, TODO(b/148406400):, UNKNOWN_TYPE, means, the, anomaly, type, is, unknown.)
 
    if (feature->has_domain() || feature->has_string_domain()) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           absl::StrCat("unspecified type: inferring the type to "
                        "be BYTES, given the domain specified.")});
      feature->set_type(tensorflow::metadata::v0::BYTES);
    } else {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           absl::StrCat("unspecified type: determine the type and "
                        "set it, rather than deprecating.")});
      // Deprecating the feature is the only possible "fix" here.
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return descriptions;
    }
  }
  if (feature->presence().min_fraction() < 0.0) {
    feature->mutable_presence()->clear_min_fraction();
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         "min_fraction should not be negative: clear is equal to zero"});
  }
  if (feature->presence().min_fraction() > 1.0) {
    feature->mutable_presence()->set_min_fraction(1.0);
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "min_fraction should not greater than 1"});
  }
  if (feature->value_count().min() < 0) {
    feature->mutable_value_count()->clear_min();
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "min should not be negative"});
  }
  if (feature->value_count().has_max() &&
      feature->value_count().max() < feature->value_count().min()) {
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "max should not be less than min"});
    feature->mutable_value_count()->set_max(feature->value_count().min());
  }
  if (!ContainsKey(AllowedFeatureTypes(feature->domain_info_case()),
                   feature->type())) {
    (/, Note, that, this, clears, the, oneof, field, domain_info.)
    ::tensorflow::data_validation::ClearDomain(feature);
    (/, TODO(b/148406400):, Give, more, detail, here.)
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "The domain does not match the type"});
  }
 
  switch (feature->domain_info_case()) {
    case Feature::kDomain:
      if (GetExistingStringDomain(feature->domain()) == nullptr) {
        // Note that this clears the oneof field domain_info.
        feature->clear_domain();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             absl::StrCat("missing domain: ", feature->domain())});
      }
      break;
    case tensorflow::metadata::v0::Feature::kBoolDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for bool domains."});
      }
      UpdateBoolDomainSelf(feature->mutable_bool_domain());
      break;
    case tensorflow::metadata::v0::Feature::kIntDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for int domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for float domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kStringDomain:
      UpdateStringDomainSelf(feature->mutable_string_domain());
      break;
    case tensorflow::metadata::v0::Feature::kStructDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for struct domains."});
      }
      break;
    case Feature::kNaturalLanguageDomain:
    case Feature::kImageDomain:
    case Feature::kMidDomain:
    case Feature::kUrlDomain:
    case Feature::kTimeDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for semantic domains."});
      }
      break;
    case Feature::DOMAIN_INFO_NOT_SET:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints require domain or string domain."});
      }
      break;
    default:
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           "internal issue: unknown domain_info type"});
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
  }
 
  return descriptions;
}
 
std::vector<Description> Schema::UpdateSkewComparator(
    const FeatureStatsView& feature_stats_view) {
  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  if (feature != nullptr &&
      FeatureHasComparator(*feature, FeatureComparatorType::SKEW)) {
    return UpdateFeatureComparatorDirect(
        feature_stats_view, FeatureComparatorType::SKEW,
        GetFeatureComparator(feature, FeatureComparatorType::SKEW));
  }
  return {};
}
 
void Schema::ClearStringDomain(const string& domain_name) {
  ClearStringDomainHelper(domain_name, schema_.mutable_feature());
  RemoveIf(schema_.mutable_string_domain(),
           [domain_name](const StringDomain* string_domain) {
             return (string_domain->name() == domain_name);
           });
}
 
std::vector<Description> Schema::UpdateFeatureInternal(
    const Updater& updater, const FeatureStatsView& view, Feature* feature) {
  std::vector<Description> descriptions = UpdateFeatureSelf(feature);
 
  // feature can be deprecated inside of UpdateFeatureSelf.
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }
 
  // This is to cover the rare case where there is actually no examples with
  // this feature, but there is still a dataset_stats object.
  const bool feature_missing = view.GetNumPresent() == 0;
 
  // If the feature is missing, but should be present, create an anomaly.
  // Otherwise, return without checking anything else.
  if (feature_missing) {
    if (IsExistenceRequired(*feature, view.environment())) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_TYPE_NOT_PRESENT,
           "Column dropped", "The feature was not present in any examples."});
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return descriptions;
    } else {
      return descriptions;
    }
  }
 
  // If the feature is present in the dataset_stats and the schema, but is
  // excluded from the environment of the dataset_stats, then add it to that
  // environment.
  if (!feature_missing &&
      get_ipython().system('IsFeatureInEnvironment(*feature, view.environment())) {')
    (/, environment, must, be, specified, here,, otherwise, all, features, would, be)
    (/, present.)
    CHECK(view.environment());
    const string view_environment = *view.environment();
    if (ContainsValue(feature->not_in_environment(), view_environment)) {
      RemoveIf(feature->mutable_not_in_environment(),
               [view_environment](const string* other) {
                 return *other == view_environment;
               });
    }
    (/, Even, if, we, remove, the, feature, from, not, in, environment,, we, may, need, to)
    (/, add, it, to, in_environment.)
    if (!IsFeatureInEnvironment(*feature, view.environment())) {
      feature->add_in_environment(view_environment);
    }
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN,
         "Column missing in environment",
         absl::StrCat("New column ", view.GetPath().Serialize(),
                      " found in data but not in the "
                      "environment ",
                      view_environment, " in the schema.")});
  }
 
  auto add_to_descriptions =
      [&descriptions](const std::vector<Description>& other_descriptions) {
        descriptions.insert(descriptions.end(), other_descriptions.begin(),
                            other_descriptions.end());
      };
 
  (/, Clear, domain_info, if, clear_field, is, set.)
  (/, Either, way,, append, descriptions.)
  auto handle_update_summary = [&descriptions,
                                feature](const UpdateSummary& update_summary) {
    descriptions.insert(descriptions.end(), update_summary.descriptions.begin(),
                        update_summary.descriptions.end());
    if (update_summary.clear_field) {
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
    }
  };
 
  if (feature->has_value_count()) {
    add_to_descriptions(::tensorflow::data_validation::UpdateValueCount(
        view, feature->mutable_value_count()));
  }
 
  if (feature->has_presence()) {
    add_to_descriptions(::tensorflow::data_validation::UpdatePresence(
        view, feature->mutable_presence()));
  }
 
  if (view.GetFeatureType() != feature->type()) {
    // Basically, deprecate the feature. The rest is just getting a meaningful
    // message out.
    ::tensorflow::data_validation::DeprecateFeature(feature);
    const ::tensorflow::protobuf::EnumValueDescriptor* descriptor =
        tensorflow::metadata::v0::FeatureNameStatistics_Type_descriptor()
            ->FindValueByNumber(view.type());
    string data_type_name = (descriptor == nullptr)
                                ? absl::StrCat("unknown(", view.type(), ")")
                                : descriptor->name();
 
    const ::tensorflow::protobuf::EnumValueDescriptor* schema_descriptor =
        tensorflow::metadata::v0::FeatureType_descriptor()->FindValueByNumber(
            feature->type());
    string schema_type_name =
        (schema_descriptor == nullptr)
            get_ipython().run_line_magic('pinfo', 'absl')
            : schema_descriptor->name();
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat("Expected data of type: ", schema_type_name, " but got ",
                      data_type_name)});
  }
 
  if (view.type() == FeatureNameStatistics::BYTES &&
      !ContainsKey(
          std::set<Feature::DomainInfoCase>(
              {Feature::DOMAIN_INFO_NOT_SET, Feature::kNaturalLanguageDomain,
               Feature::kImageDomain, Feature::kUrlDomain}),
          feature->domain_info_case())) {
    (/, Note, that, this, clears, the, oneof, field, domain_info.)
    ::tensorflow::data_validation::ClearDomain(feature);
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat("Data is marked as BYTES with incompatible "
                      "domain_info: ",
                      feature->DebugString())});
  }
  switch (feature->domain_info_case()) {
    case Feature::kDomain: {
      UpdateSummary update_summary =
          ::tensorflow::data_validation::UpdateStringDomain(
              updater, view,
              ::tensorflow::data_validation::GetMaxOffDomain(
                  feature->distribution_constraints()),
              CHECK_NOTNULL(GetExistingStringDomain(feature->domain())));
 
      descriptions.insert(descriptions.end(),
                          update_summary.descriptions.begin(),
                          update_summary.descriptions.end());
      if (update_summary.clear_field) {
        // Note that this clears the oneof field domain_info.
        const string domain = feature->domain();
        ClearStringDomain(domain);
      }
    }
 
    break;
    case Feature::kBoolDomain:
      add_to_descriptions(
          ::tensorflow::data_validation::UpdateBoolDomain(view, feature));
      break;
    case Feature::kIntDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateIntDomain(
          view, feature->mutable_int_domain()));
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateFloatDomain(
          view, feature->mutable_float_domain()));
      break;
    case tensorflow::metadata::v0::Feature::kStringDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateStringDomain(
          updater, view,
          ::tensorflow::data_validation::GetMaxOffDomain(
              feature->distribution_constraints()),
          feature->mutable_string_domain()));
      break;
    case Feature::kNaturalLanguageDomain:
    case Feature::kImageDomain:
    case Feature::kMidDomain:
    case Feature::kUrlDomain:
    case Feature::kTimeDomain:
      // Updating existing semantic domains is not supported currently.
      break;
    case Feature::kStructDomain:
      // struct_domain is handled recursively.
      break;
    case Feature::DOMAIN_INFO_NOT_SET:
      // If the domain_info is not set, it is safe to try best-effort
      // semantic type update.
      if (BestEffortUpdateCustomDomain(view.custom_stats(), feature)) {
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::SEMANTIC_DOMAIN_UPDATE,
             "Updated semantic domain",
             absl::StrCat("Updated semantic domain for feature: ",
                          feature->name())});
      }
      break;
    default:
      // In theory, default should have already been handled inside
      // UpdateFeatureSelf().
      LOG(ERROR) << "Internal error: unknown domains should be cleared inside "
                    "UpdateFeatureSelf.";
      DCHECK(false);
  }
 
  const std::vector<FeatureComparatorType> all_comparator_types = {
      FeatureComparatorType::DRIFT, FeatureComparatorType::SKEW};
  (/, Handle, comparators, here.)
  for (const auto& comparator_type : all_comparator_types) {
    if (FeatureHasComparator(*feature, comparator_type)) {
      add_to_descriptions(UpdateFeatureComparatorDirect(
          view, comparator_type,
          GetFeatureComparator(feature, comparator_type)));
    }
  }
 
  return descriptions;
}
 
std::vector<Description> Schema::UpdateSparseFeature(
    const FeatureStatsView& view, SparseFeature* sparse_feature) {
  std::vector<Description> descriptions;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the sparse_feature_stats_generator.
    if (stat_name == kMissingSparseValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature")});
    } else if (stat_name == kMissingSparseIndex) {
      for (const auto& bucket : custom_stat.rank_histogram().buckets()) {
        // This represents the index_feature name of this sparse feature.
        const string& index_feature_name = bucket.label();
        const int freq = bucket.sample_count();
        if (freq != 0) {
          descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::
                   SPARSE_FEATURE_MISSING_INDEX,
               "Missing index feature",
               absl::StrCat("Found ", freq, " examples missing index feature: ",
                            index_feature_name)});
        }
      }
    } else if (stat_name == kMaxLengthDiff || stat_name == kMinLengthDiff) {
      for (const auto& bucket : custom_stat.rank_histogram().buckets()) {
        if (bucket.sample_count() != 0) {
          // This represents the index_feature name of this sparse feature.
          const string& index_feature_name = bucket.label();
          const int difference = bucket.sample_count();
          descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::
                   SPARSE_FEATURE_LENGTH_MISMATCH,
               "Length mismatch between value and index feature",
               absl::StrCat(
                   "Mismatch between index feature: ", index_feature_name,
                   " and value column, with ", stat_name, " = ", difference)});
        }
      }
    }
    // Intentionally not generating anomalies for unknown custom stats for
    // forward compatibility.
  }
  if (!descriptions.empty()) {
    ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
  }
  return descriptions;
}
 
std::vector<Description> Schema::UpdateWeightedFeature(
    const FeatureStatsView& view, WeightedFeature* weighted_feature) {
  std::vector<Description> descriptions;
  int min_weight_length_diff = 0;
  int max_weight_length_diff = 0;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the weighted_feature_stats_generator.
    if (stat_name == kMissingWeightedValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::
               WEIGHTED_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature.")});
    } else if (stat_name == kMissingWeight && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::
               WEIGHTED_FEATURE_MISSING_WEIGHT,
           "Missing weight feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing weight feature.")});
    } else if (stat_name == kMinWeightLengthDiff && custom_stat.num() != 0) {
      min_weight_length_diff = custom_stat.num();
    } else if (stat_name == kMaxWeightLengthDiff && custom_stat.num() != 0) {
      max_weight_length_diff = custom_stat.num();
    }
  }
  if (min_weight_length_diff != 0 || max_weight_length_diff != 0) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::
             WEIGHTED_FEATURE_LENGTH_MISMATCH,
         "Length mismatch between value and weight feature",
         absl::StrCat("Mismatch between weight and value feature with ",
                      kMinWeightLengthDiff, " = ", min_weight_length_diff,
                      " and ", kMaxWeightLengthDiff, " = ",
                      max_weight_length_diff, ".")});
  }
  if (!descriptions.empty()) {
    ::tensorflow::data_validation::DeprecateWeightedFeature(weighted_feature);
  }
  return descriptions;
}
 
std::vector<Description> Schema::UpdateDatasetConstraints(
    const DatasetStatsView& dataset_stats_view) {
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::DatasetConstraints* dataset_constraints =
      GetExistingDatasetConstraints();
  if (dataset_constraints != nullptr) {
    const std::vector<DatasetComparatorType> all_comparator_types = {
        DatasetComparatorType::DRIFT, DatasetComparatorType::VERSION};
    for (const auto& comparator_type : all_comparator_types) {
      if (DatasetConstraintsHasComparator(*dataset_constraints,
                                          comparator_type)) {
        std::vector<Description> comparator_description_updates =
            UpdateNumExamplesComparatorDirect(
                dataset_stats_view, comparator_type,
                GetNumExamplesComparator(dataset_constraints, comparator_type));
        descriptions.insert(descriptions.end(),
                            comparator_description_updates.begin(),
                            comparator_description_updates.end());
      }
    }
    if (dataset_constraints->has_min_examples_count()) {
      std::vector<Description> min_examples_description_updates =
          UpdateMinExamplesCount(dataset_stats_view, dataset_constraints);
      descriptions.insert(descriptions.end(),
                          min_examples_description_updates.begin(),
                          min_examples_description_updates.end());
    }
  }
  return descriptions;
}
 
}  // namespace data_validation
}  // namespace tensorflow
 


# In[ ]:


*(Copyright, 2018, Google, LLC)
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    https://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
 
#include "tensorflow_data_validation/anomalies/schema.h"
 
#include <map>
#include <memory>
#include <set>
#include <string>
 
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow_data_validation/anomalies/bool_domain_util.h"
#include "tensorflow_data_validation/anomalies/custom_domain_util.h"
#include "tensorflow_data_validation/anomalies/dataset_constraints_util.h"
#include "tensorflow_data_validation/anomalies/feature_util.h"
#include "tensorflow_data_validation/anomalies/float_domain_util.h"
#include "tensorflow_data_validation/anomalies/int_domain_util.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow_data_validation/anomalies/map_util.h"
#include "tensorflow_data_validation/anomalies/path.h"
#include "tensorflow_data_validation/anomalies/schema_util.h"
#include "tensorflow_data_validation/anomalies/statistics_view.h"
#include "tensorflow_data_validation/anomalies/string_domain_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"
 
namespace tensorflow {
namespace data_validation {
namespace {
using ::absl::optional;
using ::tensorflow::Status;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::metadata::v0::Feature;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::SparseFeature;
using ::tensorflow::metadata::v0::StringDomain;
using ::tensorflow::metadata::v0::WeightedFeature;
using PathProto = ::tensorflow::metadata::v0::Path;
 
constexpr char kTrainingServingSkew[] = "Training/Serving skew";
 
// LINT.IfChange(sparse_feature_custom_stat_names)
static constexpr char kMissingSparseValue[] = "missing_value";
static constexpr char kMissingSparseIndex[] = "missing_index";
static constexpr char kMaxLengthDiff[] = "max_length_diff";
static constexpr char kMinLengthDiff[] = "min_length_diff";
// LINT.ThenChange(../statistics/generators/sparse_feature_stats_generator.py:custom_stat_names)
 
// LINT.IfChange(weighted_feature_custom_stat_names)
static constexpr char kMissingWeightedValue[] = "missing_value";
static constexpr char kMissingWeight[] = "missing_weight";
static constexpr char kMaxWeightLengthDiff[] = "max_weight_length_diff";
static constexpr char kMinWeightLengthDiff[] = "min_weight_length_diff";
// LINT.ThenChange(../statistics/generators/weighted_feature_stats_generator.py:custom_stat_names)
 
template <typename Container>
bool ContainsValue(const Container& a, const string& value) {
  return absl::c_find(a, value) != a.end();
}
 
std::set<tensorflow::metadata::v0::FeatureType> AllowedFeatureTypes(
    Feature::DomainInfoCase domain_info_case) {
  switch (domain_info_case) {
    case Feature::kDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kBoolDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES,
              tensorflow::metadata::v0::FLOAT};
    case Feature::kIntDomain:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::kFloatDomain:
      return {tensorflow::metadata::v0::FLOAT, tensorflow::metadata::v0::BYTES};
    case Feature::kStringDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kStructDomain:
      return {tensorflow::metadata::v0::STRUCT};
    case Feature::kNaturalLanguageDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kImageDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kMidDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kUrlDomain:
      return {tensorflow::metadata::v0::BYTES};
    case Feature::kTimeDomain:
      // Consider also supporting time as floats.
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::BYTES};
    case Feature::DOMAIN_INFO_NOT_SET:
      ABSL_FALLTHROUGH_INTENDED;
    default:
      return {tensorflow::metadata::v0::INT, tensorflow::metadata::v0::FLOAT,
              tensorflow::metadata::v0::BYTES,
              tensorflow::metadata::v0::STRUCT};
  }
}
 
// Remove all elements from the input array for which the input predicate
// pred is true. Returns number of erased elements.
template <typename T, typename Predicate>
int RemoveIf(::tensorflow::protobuf::RepeatedPtrField<T>* array,
             const Predicate& pred) {
  int i = 0, end = array->size();
  while (i < end && !pred(&array->Get(i))) ++i;
 
  if (i == end) return 0;
 
  (/, 'i', is, positioned, at, first, element, to, be, removed.)
  for (int j = i + 1; j < end; ++j) {
    if (!pred(&array->Get(j))) array->SwapElements(j, i++);
  }
 
  array->DeleteSubrange(i, end - i);
  return end - i;
}
 
Feature* GetExistingFeatureHelper(
    const string& last_part,
    tensorflow::protobuf::RepeatedPtrField<Feature>* features) {
  for (tensorflow::metadata::v0::Feature& feature : *features) {
    if (feature.name() == last_part) {
      return &feature;
    }
  }
  return nullptr;
}
 
void ClearStringDomainHelper(
    const string& domain_name,
    tensorflow::protobuf::RepeatedPtrField<Feature>* features) {
  for (tensorflow::metadata::v0::Feature& feature : *features) {
    if (feature.domain() == domain_name) {
      ::tensorflow::data_validation::ClearDomain(&feature);
    }
    if (feature.has_struct_domain()) {
      ClearStringDomainHelper(
          domain_name, feature.mutable_struct_domain()->mutable_feature());
    }
  }
}
 
SparseFeature* GetExistingSparseFeatureHelper(
    const string& name,
    tensorflow::protobuf::RepeatedPtrField<
        tensorflow::metadata::v0::SparseFeature>* sparse_features) {
  for (SparseFeature& sparse_feature : *sparse_features) {
    if (sparse_feature.name() == name) {
      return &sparse_feature;
    }
  }
  return nullptr;
}
 
(/, absl::nullopt, is, the, set, of, all, paths.)
bool ContainsPath(const absl::optional<std::set<Path>>& paths_to_consider,
                  const Path& path) {
  if (!paths_to_consider) {
    return true;
  }
  return ContainsKey(*paths_to_consider, path);
}
 
}  // namespace
 
Status Schema::Init(const tensorflow::metadata::v0::Schema& input) {
  if (!IsEmpty()) {
    return InvalidArgument("Schema is not empty when Init() called.");
  }
  schema_ = input;
  return Status::OK();
}
 
Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config) {
  return Update(dataset_stats, Updater(config), absl::nullopt);
}
 
Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const FeatureStatisticsToProtoConfig& config,
                      const std::vector<Path>& paths_to_consider) {
  return Update(
      dataset_stats, Updater(config),
      std::set<Path>(paths_to_consider.begin(), paths_to_consider.end()));
}
 
tensorflow::Status Schema::UpdateFeature(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    std::vector<Description>* descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;
 
  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  SparseFeature* sparse_feature =
      GetExistingSparseFeature(feature_stats_view.GetPath());
  WeightedFeature* weighted_feature =
      GetExistingWeightedFeature(feature_stats_view.GetPath());
  if (weighted_feature != nullptr) {
    if ((feature != nullptr || sparse_feature != nullptr) &&
        !::tensorflow::data_validation::WeightedFeatureIsDeprecated(
            *weighted_feature)) {
      descriptions->push_back({tensorflow::metadata::v0::AnomalyInfo::
                                   WEIGHTED_FEATURE_NAME_COLLISION,
                               "Weighted feature name collision",
                               "Weighted feature name collision."});
      ::tensorflow::data_validation::DeprecateWeightedFeature(weighted_feature);
      if (feature != nullptr) {
        ::tensorflow::data_validation::DeprecateFeature(feature);
      }
      if (sparse_feature != nullptr) {
        ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      }
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    } else {
      *descriptions =
          UpdateWeightedFeature(feature_stats_view, weighted_feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    }
  }
 
  if (sparse_feature != nullptr &&
      !::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature)) {
    if (feature != nullptr &&
        !::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
      descriptions->push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_NAME_COLLISION,
           "Sparse feature name collision", "Sparse feature name collision."});
      ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
      ::tensorflow::data_validation::DeprecateFeature(feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    } else {
      *descriptions = UpdateSparseFeature(feature_stats_view, sparse_feature);
      updater.UpdateSeverityForAnomaly(*descriptions, severity);
      return Status::OK();
    }
  }
 
  if (feature != nullptr) {
    *descriptions = UpdateFeatureInternal(updater, feature_stats_view, feature);
    updater.UpdateSeverityForAnomaly(*descriptions, severity);
    return Status::OK();
  } else {
    const Description description = {
        tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN, "New column",
        "New column (column in data but not in schema)"};
    *descriptions = {description};
    updater.UpdateSeverityForAnomaly(*descriptions, severity);
    return updater.CreateColumn(feature_stats_view, this, severity);
  }
  return Status::OK();
}
 
bool Schema::FeatureIsDeprecated(const Path& path) {
  Feature* feature = GetExistingFeature(path);
  if (feature == nullptr) {
    SparseFeature* sparse_feature = GetExistingSparseFeature(path);
    if (sparse_feature != nullptr) {
      return ::tensorflow::data_validation::SparseFeatureIsDeprecated(
          *sparse_feature);
    }
    // Here, the result is undefined.
    return false;
  }
  return ::tensorflow::data_validation::FeatureIsDeprecated(*feature);
}
 
void Schema::DeprecateFeature(const Path& path) {
  ::tensorflow::data_validation::DeprecateFeature(
      CHECK_NOTNULL(GetExistingFeature(path)));
}
 
Status Schema::UpdateRecursively(
    const Updater& updater, const FeatureStatsView& feature_stats_view,
    const absl::optional<std::set<Path>>& paths_to_consider,
    std::vector<Description>* descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) {
  *severity = tensorflow::metadata::v0::AnomalyInfo::UNKNOWN;
  if (!ContainsPath(paths_to_consider, feature_stats_view.GetPath())) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFeature(updater, feature_stats_view, descriptions, severity));
  if (!FeatureIsDeprecated(feature_stats_view.GetPath())) {
    for (const FeatureStatsView& child : feature_stats_view.GetChildren()) {
      std::vector<Description> child_descriptions;
      tensorflow::metadata::v0::AnomalyInfo::Severity child_severity;
      TF_RETURN_IF_ERROR(UpdateRecursively(updater, child, paths_to_consider,
                                           &child_descriptions,
                                           &child_severity));
      descriptions->insert(descriptions->end(), child_descriptions.begin(),
                           child_descriptions.end());
      *severity = MaxSeverity(child_severity, *severity);
    }
  }
  updater.UpdateSeverityForAnomaly(*descriptions, severity);
  return Status::OK();
}
 
Schema::Updater::Updater(const FeatureStatisticsToProtoConfig& config)
    : config_(config),
      columns_to_ignore_(config.column_to_ignore().begin(),
                         config.column_to_ignore().end()) {
  for (const ColumnConstraint& constraint : config.column_constraint()) {
    for (const PathProto& column_path : constraint.column_path()) {
      grouped_enums_[Path(column_path)] = constraint.enum_name();
    }
  }
}
 
(/, Sets, the, severity, based, on, anomaly, descriptions,, possibly, using, severity)
(/, overrides.)
void Schema::Updater::UpdateSeverityForAnomaly(
    const std::vector<Description>& descriptions,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  for (const auto& description : descriptions) {
    // By default, all anomalies are ERROR level.
    tensorflow::metadata::v0::AnomalyInfo::Severity severity_for_anomaly =
        tensorflow::metadata::v0::AnomalyInfo::ERROR;
 
    if (config_.new_features_are_warnings() &&
        (description.type ==
         tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN)) {
      LOG(WARNING) << "new_features_are_warnings is deprecated. Use "
                      "severity_overrides";
      severity_for_anomaly = tensorflow::metadata::v0::AnomalyInfo::WARNING;
    }
    for (const auto& severity_override : config_.severity_overrides()) {
      if (severity_override.type() == description.type) {
        severity_for_anomaly = severity_override.severity();
      }
    }
    *severity = MaxSeverity(*severity, severity_for_anomaly);
  }
}
 
Status Schema::Updater::CreateColumn(
    const FeatureStatsView& feature_stats_view, Schema* schema,
    tensorflow::metadata::v0::AnomalyInfo::Severity* severity) const {
  if (schema->GetExistingFeature(feature_stats_view.GetPath()) != nullptr) {
    return InvalidArgument("Schema already contains \"",
                           feature_stats_view.GetPath().Serialize(), "\".");
  }
 
  Feature* feature = schema->GetNewFeature(feature_stats_view.GetPath());
 
  feature->set_type(feature_stats_view.GetFeatureType());
  InitValueCountAndPresence(feature_stats_view, feature);
  if (ContainsKey(columns_to_ignore_,
                  feature_stats_view.GetPath().Serialize())) {
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return Status::OK();
  }
 
  if (BestEffortUpdateCustomDomain(feature_stats_view.custom_stats(),
                                   feature)) {
    return Status::OK();
  } else if (ContainsKey(grouped_enums_, feature_stats_view.GetPath())) {
    const string& enum_name = grouped_enums_.at(feature_stats_view.GetPath());
    StringDomain* result = schema->GetExistingStringDomain(enum_name);
    if (result == nullptr) {
      result = schema->GetNewStringDomain(enum_name);
    }
    UpdateStringDomain(*this, feature_stats_view, 0, result);
    return Status::OK();
  } else if (feature_stats_view.HasInvalidUTF8Strings() ||
             feature_stats_view.type() == FeatureNameStatistics::BYTES) {
    // If there are invalid UTF8 strings, or the field should not be further
    // interpreted, add no domain info.
    return Status::OK();
  } else if (IsBoolDomainCandidate(feature_stats_view)) {
    *feature->mutable_bool_domain() = BoolDomainFromStats(feature_stats_view);
    return Status::OK();
  } else if (IsIntDomainCandidate(feature_stats_view)) {
    // By default don't set any values.
    feature->mutable_int_domain();
    return Status::OK();
  } else if (IsStringDomainCandidate(feature_stats_view,
                                     config_.enum_threshold())) {
    StringDomain* string_domain =
        schema->GetNewStringDomain(feature_stats_view.GetPath().Serialize());
    UpdateStringDomain(*this, feature_stats_view, 0, string_domain);
    *feature->mutable_domain() = string_domain->name();
    return Status::OK();
  } else {
    (/, No, domain, info, for, this, field.)
    return Status::OK();
  }
}
 
(/, Returns, true, if, there, is, a, limit, on, the, size, of, a, string, domain, and, it)
(/, should, be, deleted.)
bool Schema::Updater::string_domain_too_big(int size) const {
  return config_.has_enum_delete_threshold() &&
         config_.enum_delete_threshold() <= size;
}
 
bool Schema::IsEmpty() const {
  return schema_.feature().empty() && schema_.string_domain().empty();
}
 
void Schema::Clear() { schema_.Clear(); }
 
StringDomain* Schema::GetNewStringDomain(const string& candidate_name) {
  std::set<string> names;
  for (const StringDomain& string_domain : schema_.string_domain()) {
    names.insert(string_domain.name());
  }
  string new_name = candidate_name;
  int index = 1;
  while (ContainsKey(names, new_name)) {
    ++index;
    new_name = absl::StrCat(candidate_name, index);
  }
  StringDomain* result = schema_.add_string_domain();
  *result->mutable_name() = new_name;
  return result;
}
 
StringDomain* Schema::GetExistingStringDomain(const string& name) {
  for (int i = 0; i < schema_.string_domain_size(); ++i) {
    StringDomain* possible = schema_.mutable_string_domain(i);
    if (possible->name() == name) {
      return possible;
    }
  }
 
  // If there is no match, return nullptr.
  return nullptr;
}
 
std::vector<std::set<string>> Schema::SimilarEnumTypes(
    const EnumsSimilarConfig& config) const {
  std::vector<bool> used(schema_.string_domain_size(), false);
  std::vector<std::set<string>> result;
  for (int index_a = 0; index_a < schema_.string_domain_size(); ++index_a) {
    if (!used[index_a]) {
      const StringDomain& string_domain_a = schema_.string_domain(index_a);
      std::set<string> similar;
      for (int index_b = index_a + 1; index_b < schema_.string_domain_size();
           ++index_b) {
        if (!used[index_b]) {
          const StringDomain& string_domain_b = schema_.string_domain(index_b);
          if (IsSimilarStringDomain(string_domain_a, string_domain_b, config)) {
            similar.insert(string_domain_b.name());
          }
        }
      }
      if (!similar.empty()) {
        similar.insert(string_domain_a.name());
        result.push_back(similar);
      }
    }
  }
  return result;
}
 
std::vector<Path> Schema::GetAllRequiredFeatures(
    const Path& prefix,
    const tensorflow::protobuf::RepeatedPtrField<Feature>& features,
    const absl::optional<string>& environment) const {
  // This recursively walks through the structure. Sometimes, a feature is
  // not required because its parent is deprecated.
  std::vector<Path> result;
  for (const Feature& feature : features) {
    const Path child_path = prefix.GetChild(feature.name());
    if (IsExistenceRequired(feature, environment)) {
      result.push_back(child_path);
    }
    // There is an odd semantics here. Here, if a child feature is required,
    // but the parent is not, we could have an anomaly for the missing child
    // feature, even though it is the parent that is actually missing.
    if (!::tensorflow::data_validation::FeatureIsDeprecated(feature)) {
      std::vector<Path> descendants = GetAllRequiredFeatures(
          child_path, feature.struct_domain().feature(), environment);
      result.insert(result.end(), descendants.begin(), descendants.end());
    }
  }
  return result;
}
 
std::vector<Path> Schema::GetMissingPaths(
    const DatasetStatsView& dataset_stats) {
  std::set<Path> paths_present;
  for (const FeatureStatsView& feature_stats_view : dataset_stats.features()) {
    paths_present.insert(feature_stats_view.GetPath());
  }
  std::vector<Path> paths_absent;
 
  for (const Path& path : GetAllRequiredFeatures(Path(), schema_.feature(),
                                                 dataset_stats.environment())) {
    if (!ContainsKey(paths_present, path)) {
      paths_absent.push_back(path);
    }
  }
  return paths_absent;
}
 
(/, TODO(b/148406484):, currently,, only, looks, at, top-level, features.)
(/, Make, this, include, lower, level, features, as, well.)
(/, See, also, b/114757721.)
std::map<string, std::set<Path>> Schema::EnumNameToPaths() const {
  std::map<string, std::set<Path>> result;
  for (const Feature& feature : schema_.feature()) {
    if (feature.has_domain()) {
      result[feature.domain()].insert(Path({feature.name()}));
    }
  }
  return result;
}
 
Status Schema::Update(const DatasetStatsView& dataset_stats,
                      const Updater& updater,
                      const absl::optional<std::set<Path>>& paths_to_consider) {
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::AnomalyInfo::Severity severity;
 
  for (const auto& feature_stats_view : dataset_stats.GetRootFeatures()) {
    TF_RETURN_IF_ERROR(UpdateRecursively(updater, feature_stats_view,
                                         paths_to_consider, &descriptions,
                                         &severity));
  }
  for (const Path& missing_path : GetMissingPaths(dataset_stats)) {
    if (ContainsPath(paths_to_consider, missing_path)) {
      DeprecateFeature(missing_path);
    }
  }
  return Status::OK();
}
 
(/, TODO(b/114757721):, expose, this.)
Status Schema::GetRelatedEnums(const DatasetStatsView& dataset_stats,
                               FeatureStatisticsToProtoConfig* config) {
  Schema schema;
  TF_RETURN_IF_ERROR(schema.Update(dataset_stats, *config));
 
  std::vector<std::set<string>> similar_enums =
      schema.SimilarEnumTypes(config->enums_similar_config());
  // Map the enum names to the paths.
  const std::map<string, std::set<Path>> enum_name_to_paths =
      schema.EnumNameToPaths();
  for (const std::set<string>& set : similar_enums) {
    if (set.empty()) {
      return Internal("Schema::SimilarEnumTypes returned an empty set.");
    }
    ColumnConstraint* column_constraint = config->add_column_constraint();
    for (const string& enum_name : set) {
      if (ContainsKey(enum_name_to_paths, enum_name)) {
        for (const auto& column : enum_name_to_paths.at(enum_name)) {
          *column_constraint->add_column_path() = column.AsProto();
        }
      }
    }
    // Choose the shortest name for the enum.
    string best_name = *set.begin();
    for (const string& current_name : set) {
      if (current_name.size() < best_name.size()) {
        best_name = current_name;
      }
    }
    *column_constraint->mutable_enum_name() = best_name;
  }
  return Status::OK();
}
 
tensorflow::metadata::v0::Schema Schema::GetSchema() const { return schema_; }
 
bool Schema::FeatureExists(const Path& path) {
  return GetExistingFeature(path) != nullptr ||
         GetExistingSparseFeature(path) != nullptr ||
         GetExistingWeightedFeature(path) != nullptr;
}
 
Feature* Schema::GetExistingFeature(const Path& path) {
  if (path.size() == 1) {
    return GetExistingFeatureHelper(path.last_step(),
                                    schema_.mutable_feature());
  } else {
    Path parent = path.GetParent();
    Feature* parent_feature = GetExistingFeature(parent);
    if (parent_feature == nullptr) {
      return nullptr;
    }
    if (!parent_feature->has_struct_domain()) {
      return nullptr;
    }
    return GetExistingFeatureHelper(
        path.last_step(),
        parent_feature->mutable_struct_domain()->mutable_feature());
  }
  return nullptr;
}
 
SparseFeature* Schema::GetExistingSparseFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() == 1) {
    return GetExistingSparseFeatureHelper(path.last_step(),
                                          schema_.mutable_sparse_feature());
  } else {
    Feature* parent_feature = GetExistingFeature(path.GetParent());
    if (parent_feature == nullptr) {
      return nullptr;
    }
    if (!parent_feature->has_struct_domain()) {
      return nullptr;
    }
    return GetExistingSparseFeatureHelper(
        path.last_step(),
        parent_feature->mutable_struct_domain()->mutable_sparse_feature());
  }
}
 
WeightedFeature* Schema::GetExistingWeightedFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() != 1) {
    // Weighted features are always top-level features with single-step paths.
    return nullptr;
  }
  auto name = path.last_step();
  for (WeightedFeature& weighted_feature :
       *schema_.mutable_weighted_feature()) {
    if (weighted_feature.name() == name) {
      return &weighted_feature;
    }
  }
  return nullptr;
}
 
Feature* Schema::GetNewFeature(const Path& path) {
  CHECK(!path.empty());
  if (path.size() > 1) {
    Path parent = path.GetParent();
    Feature* parent_feature = CHECK_NOTNULL(GetExistingFeature(parent));
    Feature* result = parent_feature->mutable_struct_domain()->add_feature();
    *result->mutable_name() = path.last_step();
    return result;
  } else {
    Feature* result = schema_.add_feature();
    *result->mutable_name() = path.last_step();
    return result;
  }
}
 
::tensorflow::metadata::v0::DatasetConstraints*
Schema::GetExistingDatasetConstraints() {
  if (schema_.has_dataset_constraints()) {
    return schema_.mutable_dataset_constraints();
  }
  return nullptr;
}
 
bool Schema::IsFeatureInEnvironment(
    const Feature& feature, const absl::optional<string>& environment) const {
  if (environment) {
    if (ContainsValue(feature.in_environment(), *environment)) {
      return true;
    }
    if (ContainsValue(feature.not_in_environment(), *environment)) {
      return false;
    }
    if (ContainsValue(schema_.default_environment(), *environment)) {
      return true;
    }
    return false;
  }
  // If environment is not set, then the feature is considered in the
  // environment by default.
  return true;
}
 
bool Schema::IsExistenceRequired(
    const Feature& feature, const absl::optional<string>& environment) const {
  if (::tensorflow::data_validation::FeatureIsDeprecated(feature)) {
    return false;
  }
  if (feature.presence().min_count() <= 0 &&
      feature.presence().min_fraction() <= 0.0) {
    return false;
  }
  // If a feature is in the environment, it is required.
  return IsFeatureInEnvironment(feature, environment);
}
 
(/, TODO(b/148406400):, Switch, AnomalyInfo::Type, from, UNKNOWN_TYPE.)
(/, TODO(b/148406994):, Handle, missing, FeatureType, more, elegantly,, inferring, it)
(/, when, necessary.)
std::vector<Description> Schema::UpdateFeatureSelf(Feature* feature) {
  std::vector<Description> descriptions;
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }
  if (!feature->has_name()) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat(
             "unspecified name (maybe meant to be the empty string): find "
             "name rather than deprecating.")});
    (/, Deprecating, the, feature, is, the, only, possible, "fix", here.)
    ::tensorflow::data_validation::DeprecateFeature(feature);
    return descriptions;
  }
 
  if (!feature->has_type()) {
    (/, TODO(b/148406400):, UNKNOWN_TYPE, means, the, anomaly, type, is, unknown.)
 
    if (feature->has_domain() || feature->has_string_domain()) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           absl::StrCat("unspecified type: inferring the type to "
                        "be BYTES, given the domain specified.")});
      feature->set_type(tensorflow::metadata::v0::BYTES);
    } else {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           absl::StrCat("unspecified type: determine the type and "
                        "set it, rather than deprecating.")});
      // Deprecating the feature is the only possible "fix" here.
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return descriptions;
    }
  }
  if (feature->presence().min_fraction() < 0.0) {
    feature->mutable_presence()->clear_min_fraction();
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         "min_fraction should not be negative: clear is equal to zero"});
  }
  if (feature->presence().min_fraction() > 1.0) {
    feature->mutable_presence()->set_min_fraction(1.0);
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "min_fraction should not greater than 1"});
  }
  if (feature->value_count().min() < 0) {
    feature->mutable_value_count()->clear_min();
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "min should not be negative"});
  }
  if (feature->value_count().has_max() &&
      feature->value_count().max() < feature->value_count().min()) {
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "max should not be less than min"});
    feature->mutable_value_count()->set_max(feature->value_count().min());
  }
  if (!ContainsKey(AllowedFeatureTypes(feature->domain_info_case()),
                   feature->type())) {
    (/, Note, that, this, clears, the, oneof, field, domain_info.)
    ::tensorflow::data_validation::ClearDomain(feature);
    (/, TODO(b/148406400):, Give, more, detail, here.)
    descriptions.push_back({tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
                            "The domain does not match the type"});
  }
 
  switch (feature->domain_info_case()) {
    case Feature::kDomain:
      if (GetExistingStringDomain(feature->domain()) == nullptr) {
        // Note that this clears the oneof field domain_info.
        feature->clear_domain();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             absl::StrCat("missing domain: ", feature->domain())});
      }
      break;
    case tensorflow::metadata::v0::Feature::kBoolDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for bool domains."});
      }
      UpdateBoolDomainSelf(feature->mutable_bool_domain());
      break;
    case tensorflow::metadata::v0::Feature::kIntDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for int domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for float domains."});
      }
      break;
    case tensorflow::metadata::v0::Feature::kStringDomain:
      UpdateStringDomainSelf(feature->mutable_string_domain());
      break;
    case tensorflow::metadata::v0::Feature::kStructDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for struct domains."});
      }
      break;
    case Feature::kNaturalLanguageDomain:
    case Feature::kImageDomain:
    case Feature::kMidDomain:
    case Feature::kUrlDomain:
    case Feature::kTimeDomain:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints not supported for semantic domains."});
      }
      break;
    case Feature::DOMAIN_INFO_NOT_SET:
      if (feature->has_distribution_constraints()) {
        feature->clear_distribution_constraints();
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
             "distribution constraints require domain or string domain."});
      }
      break;
    default:
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
           "internal issue: unknown domain_info type"});
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
  }
 
  return descriptions;
}
 
std::vector<Description> Schema::UpdateSkewComparator(
    const FeatureStatsView& feature_stats_view) {
  Feature* feature = GetExistingFeature(feature_stats_view.GetPath());
  if (feature != nullptr &&
      FeatureHasComparator(*feature, FeatureComparatorType::SKEW)) {
    return UpdateFeatureComparatorDirect(
        feature_stats_view, FeatureComparatorType::SKEW,
        GetFeatureComparator(feature, FeatureComparatorType::SKEW));
  }
  return {};
}
 
void Schema::ClearStringDomain(const string& domain_name) {
  ClearStringDomainHelper(domain_name, schema_.mutable_feature());
  RemoveIf(schema_.mutable_string_domain(),
           [domain_name](const StringDomain* string_domain) {
             return (string_domain->name() == domain_name);
           });
}
 
std::vector<Description> Schema::UpdateFeatureInternal(
    const Updater& updater, const FeatureStatsView& view, Feature* feature) {
  std::vector<Description> descriptions = UpdateFeatureSelf(feature);
 
  // feature can be deprecated inside of UpdateFeatureSelf.
  if (::tensorflow::data_validation::FeatureIsDeprecated(*feature)) {
    return descriptions;
  }
 
  // This is to cover the rare case where there is actually no examples with
  // this feature, but there is still a dataset_stats object.
  const bool feature_missing = view.GetNumPresent() == 0;
 
  // If the feature is missing, but should be present, create an anomaly.
  // Otherwise, return without checking anything else.
  if (feature_missing) {
    if (IsExistenceRequired(*feature, view.environment())) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FEATURE_TYPE_NOT_PRESENT,
           "Column dropped", "The feature was not present in any examples."});
      ::tensorflow::data_validation::DeprecateFeature(feature);
      return descriptions;
    } else {
      return descriptions;
    }
  }
 
  // If the feature is present in the dataset_stats and the schema, but is
  // excluded from the environment of the dataset_stats, then add it to that
  // environment.
  if (!feature_missing &&
      get_ipython().system('IsFeatureInEnvironment(*feature, view.environment())) {')
    (/, environment, must, be, specified, here,, otherwise, all, features, would, be)
    (/, present.)
    CHECK(view.environment());
    const string view_environment = *view.environment();
    if (ContainsValue(feature->not_in_environment(), view_environment)) {
      RemoveIf(feature->mutable_not_in_environment(),
               [view_environment](const string* other) {
                 return *other == view_environment;
               });
    }
    (/, Even, if, we, remove, the, feature, from, not, in, environment,, we, may, need, to)
    (/, add, it, to, in_environment.)
    if (!IsFeatureInEnvironment(*feature, view.environment())) {
      feature->add_in_environment(view_environment);
    }
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::SCHEMA_NEW_COLUMN,
         "Column missing in environment",
         absl::StrCat("New column ", view.GetPath().Serialize(),
                      " found in data but not in the "
                      "environment ",
                      view_environment, " in the schema.")});
  }
 
  auto add_to_descriptions =
      [&descriptions](const std::vector<Description>& other_descriptions) {
        descriptions.insert(descriptions.end(), other_descriptions.begin(),
                            other_descriptions.end());
      };
 
  (/, Clear, domain_info, if, clear_field, is, set.)
  (/, Either, way,, append, descriptions.)
  auto handle_update_summary = [&descriptions,
                                feature](const UpdateSummary& update_summary) {
    descriptions.insert(descriptions.end(), update_summary.descriptions.begin(),
                        update_summary.descriptions.end());
    if (update_summary.clear_field) {
      // Note that this clears the oneof field domain_info.
      ::tensorflow::data_validation::ClearDomain(feature);
    }
  };
 
  if (feature->has_value_count()) {
    add_to_descriptions(::tensorflow::data_validation::UpdateValueCount(
        view, feature->mutable_value_count()));
  }
 
  if (feature->has_presence()) {
    add_to_descriptions(::tensorflow::data_validation::UpdatePresence(
        view, feature->mutable_presence()));
  }
 
  if (view.GetFeatureType() != feature->type()) {
    // Basically, deprecate the feature. The rest is just getting a meaningful
    // message out.
    ::tensorflow::data_validation::DeprecateFeature(feature);
    const ::tensorflow::protobuf::EnumValueDescriptor* descriptor =
        tensorflow::metadata::v0::FeatureNameStatistics_Type_descriptor()
            ->FindValueByNumber(view.type());
    string data_type_name = (descriptor == nullptr)
                                ? absl::StrCat("unknown(", view.type(), ")")
                                : descriptor->name();
 
    const ::tensorflow::protobuf::EnumValueDescriptor* schema_descriptor =
        tensorflow::metadata::v0::FeatureType_descriptor()->FindValueByNumber(
            feature->type());
    string schema_type_name =
        (schema_descriptor == nullptr)
            get_ipython().run_line_magic('pinfo', 'absl')
            : schema_descriptor->name();
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat("Expected data of type: ", schema_type_name, " but got ",
                      data_type_name)});
  }
 
  if (view.type() == FeatureNameStatistics::BYTES &&
      !ContainsKey(
          std::set<Feature::DomainInfoCase>(
              {Feature::DOMAIN_INFO_NOT_SET, Feature::kNaturalLanguageDomain,
               Feature::kImageDomain, Feature::kUrlDomain}),
          feature->domain_info_case())) {
    (/, Note, that, this, clears, the, oneof, field, domain_info.)
    ::tensorflow::data_validation::ClearDomain(feature);
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::UNKNOWN_TYPE,
         absl::StrCat("Data is marked as BYTES with incompatible "
                      "domain_info: ",
                      feature->DebugString())});
  }
  switch (feature->domain_info_case()) {
    case Feature::kDomain: {
      UpdateSummary update_summary =
          ::tensorflow::data_validation::UpdateStringDomain(
              updater, view,
              ::tensorflow::data_validation::GetMaxOffDomain(
                  feature->distribution_constraints()),
              CHECK_NOTNULL(GetExistingStringDomain(feature->domain())));
 
      descriptions.insert(descriptions.end(),
                          update_summary.descriptions.begin(),
                          update_summary.descriptions.end());
      if (update_summary.clear_field) {
        // Note that this clears the oneof field domain_info.
        const string domain = feature->domain();
        ClearStringDomain(domain);
      }
    }
 
    break;
    case Feature::kBoolDomain:
      add_to_descriptions(
          ::tensorflow::data_validation::UpdateBoolDomain(view, feature));
      break;
    case Feature::kIntDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateIntDomain(
          view, feature->mutable_int_domain()));
      break;
    case tensorflow::metadata::v0::Feature::kFloatDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateFloatDomain(
          view, feature->mutable_float_domain()));
      break;
    case tensorflow::metadata::v0::Feature::kStringDomain:
      handle_update_summary(::tensorflow::data_validation::UpdateStringDomain(
          updater, view,
          ::tensorflow::data_validation::GetMaxOffDomain(
              feature->distribution_constraints()),
          feature->mutable_string_domain()));
      break;
    case Feature::kNaturalLanguageDomain:
    case Feature::kImageDomain:
    case Feature::kMidDomain:
    case Feature::kUrlDomain:
    case Feature::kTimeDomain:
      // Updating existing semantic domains is not supported currently.
      break;
    case Feature::kStructDomain:
      // struct_domain is handled recursively.
      break;
    case Feature::DOMAIN_INFO_NOT_SET:
      // If the domain_info is not set, it is safe to try best-effort
      // semantic type update.
      if (BestEffortUpdateCustomDomain(view.custom_stats(), feature)) {
        descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::SEMANTIC_DOMAIN_UPDATE,
             "Updated semantic domain",
             absl::StrCat("Updated semantic domain for feature: ",
                          feature->name())});
      }
      break;
    default:
      // In theory, default should have already been handled inside
      // UpdateFeatureSelf().
      LOG(ERROR) << "Internal error: unknown domains should be cleared inside "
                    "UpdateFeatureSelf.";
      DCHECK(false);
  }
 
  const std::vector<FeatureComparatorType> all_comparator_types = {
      FeatureComparatorType::DRIFT, FeatureComparatorType::SKEW};
  (/, Handle, comparators, here.)
  for (const auto& comparator_type : all_comparator_types) {
    if (FeatureHasComparator(*feature, comparator_type)) {
      add_to_descriptions(UpdateFeatureComparatorDirect(
          view, comparator_type,
          GetFeatureComparator(feature, comparator_type)));
    }
  }
 
  return descriptions;
}
 
std::vector<Description> Schema::UpdateSparseFeature(
    const FeatureStatsView& view, SparseFeature* sparse_feature) {
  std::vector<Description> descriptions;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the sparse_feature_stats_generator.
    if (stat_name == kMissingSparseValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::SPARSE_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature")});
    } else if (stat_name == kMissingSparseIndex) {
      for (const auto& bucket : custom_stat.rank_histogram().buckets()) {
        // This represents the index_feature name of this sparse feature.
        const string& index_feature_name = bucket.label();
        const int freq = bucket.sample_count();
        if (freq != 0) {
          descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::
                   SPARSE_FEATURE_MISSING_INDEX,
               "Missing index feature",
               absl::StrCat("Found ", freq, " examples missing index feature: ",
                            index_feature_name)});
        }
      }
    } else if (stat_name == kMaxLengthDiff || stat_name == kMinLengthDiff) {
      for (const auto& bucket : custom_stat.rank_histogram().buckets()) {
        if (bucket.sample_count() != 0) {
          // This represents the index_feature name of this sparse feature.
          const string& index_feature_name = bucket.label();
          const int difference = bucket.sample_count();
          descriptions.push_back(
              {tensorflow::metadata::v0::AnomalyInfo::
                   SPARSE_FEATURE_LENGTH_MISMATCH,
               "Length mismatch between value and index feature",
               absl::StrCat(
                   "Mismatch between index feature: ", index_feature_name,
                   " and value column, with ", stat_name, " = ", difference)});
        }
      }
    }
    // Intentionally not generating anomalies for unknown custom stats for
    // forward compatibility.
  }
  if (!descriptions.empty()) {
    ::tensorflow::data_validation::DeprecateSparseFeature(sparse_feature);
  }
  return descriptions;
}
 
std::vector<Description> Schema::UpdateWeightedFeature(
    const FeatureStatsView& view, WeightedFeature* weighted_feature) {
  std::vector<Description> descriptions;
  int min_weight_length_diff = 0;
  int max_weight_length_diff = 0;
  for (const tensorflow::metadata::v0::CustomStatistic& custom_stat :
       view.custom_stats()) {
    const string& stat_name = custom_stat.name();
    // Stat names should be in-sync with the weighted_feature_stats_generator.
    if (stat_name == kMissingWeightedValue && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::
               WEIGHTED_FEATURE_MISSING_VALUE,
           "Missing value feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing value feature.")});
    } else if (stat_name == kMissingWeight && custom_stat.num() != 0) {
      descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::
               WEIGHTED_FEATURE_MISSING_WEIGHT,
           "Missing weight feature",
           absl::StrCat("Found ", custom_stat.num(),
                        " examples missing weight feature.")});
    } else if (stat_name == kMinWeightLengthDiff && custom_stat.num() != 0) {
      min_weight_length_diff = custom_stat.num();
    } else if (stat_name == kMaxWeightLengthDiff && custom_stat.num() != 0) {
      max_weight_length_diff = custom_stat.num();
    }
  }
  if (min_weight_length_diff != 0 || max_weight_length_diff != 0) {
    descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::
             WEIGHTED_FEATURE_LENGTH_MISMATCH,
         "Length mismatch between value and weight feature",
         absl::StrCat("Mismatch between weight and value feature with ",
                      kMinWeightLengthDiff, " = ", min_weight_length_diff,
                      " and ", kMaxWeightLengthDiff, " = ",
                      max_weight_length_diff, ".")});
  }
  if (!descriptions.empty()) {
    ::tensorflow::data_validation::DeprecateWeightedFeature(weighted_feature);
  }
  return descriptions;
}
 
std::vector<Description> Schema::UpdateDatasetConstraints(
    const DatasetStatsView& dataset_stats_view) {
  std::vector<Description> descriptions;
  tensorflow::metadata::v0::DatasetConstraints* dataset_constraints =
      GetExistingDatasetConstraints();
  if (dataset_constraints != nullptr) {
    const std::vector<DatasetComparatorType> all_comparator_types = {
        DatasetComparatorType::DRIFT, DatasetComparatorType::VERSION};
    for (const auto& comparator_type : all_comparator_types) {
      if (DatasetConstraintsHasComparator(*dataset_constraints,
                                          comparator_type)) {
        std::vector<Description> comparator_description_updates =
            UpdateNumExamplesComparatorDirect(
                dataset_stats_view, comparator_type,
                GetNumExamplesComparator(dataset_constraints, comparator_type));
        descriptions.insert(descriptions.end(),
                            comparator_description_updates.begin(),
                            comparator_description_updates.end());
      }
    }
    if (dataset_constraints->has_min_examples_count()) {
      std::vector<Description> min_examples_description_updates =
          UpdateMinExamplesCount(dataset_stats_view, dataset_constraints);
      descriptions.insert(descriptions.end(),
                          min_examples_description_updates.begin(),
                          min_examples_description_updates.end());
    }
  }
  return descriptions;
}
 
}  // namespace data_validation
}  // namespace tensorflow
 


# In[ ]:


*(Copyright, 2018, Google, LLC)
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    https://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
 
#include "tensorflow_data_validation/anomalies/float_domain_util.h"
 
#include <cmath>
#include <string>
#include <vector>
 
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow_data_validation/anomalies/internal_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_metadata/proto/v0/anomalies.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"
 
namespace tensorflow {
namespace data_validation {
namespace {
 
constexpr char kOutOfRangeValues[] = "Out-of-range values";
constexpr char kInvalidValues[] = "Invalid values";
 
using ::absl::get_if;
using ::absl::holds_alternative;
using ::absl::optional;
using ::absl::variant;
using ::tensorflow::metadata::v0::FeatureNameStatistics;
using ::tensorflow::metadata::v0::FloatDomain;
 
// A FloatIntervalResult where a byte is not a float.
typedef string ExampleStringNotFloat;
 
// An interval of floats.
struct FloatInterval {
  // Min and max values of the interval.
  float min, max;
};
 
// See GetFloatInterval
using FloatIntervalResult =
    absl::optional<variant<FloatInterval, ExampleStringNotFloat>>;
 
// Determines the range of floats represented by the feature_stats, whether
// the data is floats or strings.
// Returns nullopt if there is no data in the field or it is INT.
// Returns ExampleStringNotFloat if there is at least one string that does not
// represent a float.
// Otherwise, returns the interval.
FloatIntervalResult GetFloatInterval(const FeatureStatsView& feature_stats) {
  switch (feature_stats.type()) {
    case FeatureNameStatistics::FLOAT:
      return FloatInterval{static_cast<float>(feature_stats.num_stats().min()),
                           static_cast<float>(feature_stats.num_stats().max())};
    case FeatureNameStatistics::BYTES:
    case FeatureNameStatistics::STRING: {
      absl::optional<FloatInterval> interval;
      for (const string& str : feature_stats.GetStringValues()) {
        float value;
        if (!absl::SimpleAtof(str, &value)) {
          return str;
        }
        if (!interval) {
          interval = FloatInterval{value, value};
        }
        if (interval->min > value) {
          interval->min = value;
        }
        if (interval->max < value) {
          interval->max = value;
        }
      }
      if (interval) {
        return *interval;
      }
      return absl::nullopt;
    }
    case FeatureNameStatistics::INT:
      return absl::nullopt;
    default:
      LOG(FATAL) << "Unknown type: " << feature_stats.type();
  }
}
 
(/, Check, if, there, are, NaNs, in, a, float, feature., If, the, domain, indicates, that)
(/, NaNs, are, disallowed,, the, presence, of, a, NaN, raises, an, anomaly.)
(/, TODO(askerryryan):, Consider, merging, this, logic, with, FloatIntervalResult.)
void CheckFloatNans(const FeatureStatsView& stats,
                    UpdateSummary* update_summary,
                    tensorflow::metadata::v0::FloatDomain* float_domain) {
  bool has_nans = false;
  if (!float_domain->disallow_nan()) {
    return;
  }
  switch (stats.type()) {
    case FeatureNameStatistics::FLOAT:
      for (const auto& histogram : stats.num_stats().histograms()) {
        if (histogram.num_nan() > 0) {
          has_nans = true;
          break;
        }
      }
      break;
    case FeatureNameStatistics::STRING:
      for (const string& str : stats.GetStringValues()) {
        float value;
        if (absl::SimpleAtof(str, &value) && std::isnan(value)) {
          has_nans = true;
          break;
        }
      }
      break;
    default:
      break;
  }
  if (has_nans) {
    update_summary->descriptions.push_back(
        {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_HAS_NAN,
         kInvalidValues, absl::StrCat("Float feature has NaN values.")});
    float_domain->set_disallow_nan(false);
  }
}
}  // namespace
 
UpdateSummary UpdateFloatDomain(
    const FeatureStatsView& stats,
    tensorflow::metadata::v0::FloatDomain* float_domain) {
  UpdateSummary update_summary;
 
  CheckFloatNans(stats, &update_summary, float_domain);
 
  const FloatIntervalResult result = GetFloatInterval(stats);
  if (result) {
    const variant<FloatInterval, ExampleStringNotFloat> actual_result = *result;
    if (holds_alternative<ExampleStringNotFloat>(actual_result)) {
      update_summary.descriptions.push_back(
          {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_STRING_NOT_FLOAT,
           kInvalidValues,
           absl::StrCat(
               "String values that were not floats were found, such as \"",
               *absl::get_if<ExampleStringNotFloat>(&actual_result), "\".")});
      update_summary.clear_field = true;
      return update_summary;
    }
    if (holds_alternative<FloatInterval>(actual_result)) {
      const FloatInterval range = *absl::get_if<FloatInterval>(&actual_result);
      if (float_domain->has_min() && range.min < float_domain->min()) {
        float_domain->set_min(range.min);
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_SMALL_FLOAT,
             kOutOfRangeValues,
             absl::StrCat(
                 "Unexpectedly low values: ", absl::SixDigits(range.min), "<",
                 absl::SixDigits(float_domain->min()),
                 "(upto six significant digits)")});
      }
 
      if (float_domain->has_max() && range.max > float_domain->max()) {
        update_summary.descriptions.push_back(
            {tensorflow::metadata::v0::AnomalyInfo::FLOAT_TYPE_BIG_FLOAT,
             kOutOfRangeValues,
             absl::StrCat(
                 "Unexpectedly high value: ", absl::SixDigits(range.max), ">",
                 absl::SixDigits(float_domain->max()),
                 "(upto six significant digits)")});
        float_domain->set_max(range.max);
      }
    }
  }
  // If no interval is found, then assume everything is OK.
  return update_summary;
}
 
bool IsFloatDomainCandidate(const FeatureStatsView& feature_stats) {
  // We don't set float_domain by default unless we are trying to indicate
  (/, that, strings, are, actually, floats.)
  if (feature_stats.type() != FeatureNameStatistics::STRING ||
      feature_stats.HasInvalidUTF8Strings()) {
    return false;
  }
  const FloatIntervalResult result = GetFloatInterval(feature_stats);
  if (result) {
    // If all the examples are floats, then maybe we can make this a
    // FloatDomain.
    return holds_alternative<FloatInterval>(*result);
  }
  return false;
}
 
}  // namespace data_validation
}  // namespace tensorflow
 


# In[ ]:


licenses(["unencumbered"])  # Public Domain or MIT
 
exports_files(["LICENSE"])
 
cc_library(
    name = "jsoncpp",
    srcs = [
        "include/json/assertions.h",
        "src/lib_json/json_reader.cpp",
        "src/lib_json/json_tool.h",
        "src/lib_json/json_value.cpp",
        "src/lib_json/json_writer.cpp",
    ],
    hdrs = [
        "include/json/autolink.h",
        "include/json/config.h",
        "include/json/features.h",
        "include/json/forwards.h",
        "include/json/json.h",
        "include/json/reader.h",
        "include/json/value.h",
        "include/json/version.h",
        "include/json/writer.h",
    ],
    copts = [
        "-DJSON_USE_EXCEPTION=0",
        "-DJSON_HAS_INT64",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":private"],
)
 
cc_library(
    name = "private",
    textual_hdrs = ["src/lib_json/json_valueiterator.inl"],
)
 


# In[ ]:


pipeline {
    agent { label 'ephemeral-linux-gpu' }
    options {
        // The Build GPU stage depends on the image from the Push CPU stage
        disableConcurrentBuilds()
    }
    environment {
        GIT_COMMIT_SHORT = sh(returnStdout: true, script:"git rev-parse --short=7 HEAD").trim()
        GIT_COMMIT_SUBJECT = sh(returnStdout: true, script:"git log --format=%s -n 1 HEAD").trim()
        GIT_COMMIT_AUTHOR = sh(returnStdout: true, script:"git log --format='%an' -n 1 HEAD").trim()
        GIT_COMMIT_SUMMARY = "`<https://github.com/Kaggle/docker-python/commit/${GIT_COMMIT}|${GIT_COMMIT_SHORT}>` ${GIT_COMMIT_SUBJECT} - ${GIT_COMMIT_AUTHOR}"
    }
 
    stages {
        stage('Build') {
            steps {
                sh '''#!/bin/bash
                set -exo pipefail
                
                cd tensorflow-whl/
                ./build | ts
                '''
            }
        }
        stage('Push') {
            steps {
                sh '''#!/bin/bash
                set -exo pipefail
 
                cd tensorflow-whl/
                ./push ${GIT_BRANCH}-staging
                '''
            }
        }
    }
}
 


# In[ ]:


#!/bin/bash
set -e
 
usage() {
cat << EOF
Usage: $0 [OPTIONS]
Build new Tensorflow wheels for use in the Kaggle Docker Python base images.
 
Options:
    -c, --use-cache Use layer cache when building a new image.
EOF
}
 
CACHE_FLAG='--no-cache'
IMAGE_TAG='kaggle/python-tensorflow-whl'
 
while :; do
    case "$1" in 
        -h|--help)
            usage
            exit
            (";")
        -c|--use-cache)
            CACHE_FLAG=''
            (";")
        -?*)
            usage
            printf 'ERROR: Unknown option: %s\n' "$1" >&2
            exit
            (";")
        *)            
            break
    esac
 
    shift
done
 
readonly CACHE_FLAG
readonly IMAGE_TAG
 
set -x
docker build --rm --pull $CACHE_FLAG -t "$IMAGE_TAG" .
 


# In[ ]:



{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 14967,
      "digest": "sha256:bd26fda24a9af7ab97a0bb90b121d63b8e03130549afba03ead5bad47b210c9f"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 50382957,
         "digest": "sha256:7e2b2a5af8f65687add6d864d5841067e23bd435eb1a051be6fe1ea2384946b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 222909892,
         "digest": "sha256:59c89b5f9b0c6d94c77d4c3a42986d420aaa7575ac65fcd2c3f5968b3726abfc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 195204532,
         "digest": "sha256:4017849f9f85133e68a4125e9679775f8e46a17dcdb8c2a52bbe72d0198f5e68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1522,
         "digest": "sha256:c8b29d62979a416da925e526364a332b13f8d5f43804ae98964de2a60d47c17a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 717,
         "digest": "sha256:12004028a6a740ac35e69f489093b860968cc37b9668f65b1e2f61fd4c4ad25c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247,
         "digest": "sha256:3f09b9a53dfb03fd34e35d43694c2d38656f7431efce0e6647c47efb5f7b3137"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 408,
         "digest": "sha256:03ed58116b0cb733cc552dc89ef5ea122b6c5cf39ec467f6ad671dc0ba35db0c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 331594702,
         "digest": "sha256:7844554d9ef75bb3f1d224e166ed12561e78add339448c52a8e5679943b229f1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 112946047,
         "digest": "sha256:e24892c008d79093cf0566c69af2319f76dc59f4f631809742366d4d0f7a7866"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 425,
         "digest": "sha256:b89ff65d69ce89fe9d05fe3acf9f89046a19eaed148e80a6e167b93e6dc26423"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 5476,
         "digest": "sha256:d7a15e9b63f265b3f895e4c9f02533d105d9b277e411b93e81bb98972018d11a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1952,
         "digest": "sha256:ce5ee886afe2dbaf02011d8e98f203ff2615f18dcf43f120aaa7492d4b1310a2"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2484763037,
         "digest": "sha256:d4fb652f11a8983e448c66f2732dceefafa85dccfc48ebfb17c62ed2dcbe6507"
      }
   ]
}


# In[ ]:


*(eslint-disable, no-unused-vars, */)
const chartContent = [
  '```chart',
  ',category1,category2',
  'Jan,21,23',
  'Feb,31,17',
  '',
  'type: column',
  'title: Monthly Revenue',
  'x.title: Amount',
  'y.title: Month',
  'y.min: 1',
  'y.max: 40',
  'y.suffix: $',
  '```'
].join('\n');
 
const codeContent = [
  '```js',
  `console.log('foo')`,
  '```',
  '```javascript',
  `console.log('bar')`,
  '```',
  '```html',
  '<div id="editor"><span>baz</span></div>',
  '```',
  '```wrong',
  '[1 2 3]',
  '```',
  '```clojure',
  '[1 2 3]',
  '```'
].join('\n');
 
const tableContent = ['| @cols=2:merged |', '| --- | --- |', '| table | table2 |'].join('\n');
 
const umlContent = [
  '```uml',
  'partition Conductor {',
  '  (*) --> "Climbs on Platform"',
  '  --> === S1 ===',
  '  --> Bows',
  '}',
  '',
  'partition Audience #LightSkyBlue {',
  '  === S1 === --> Applauds',
  '}',
  '',
  'partition Conductor {',
  '  Bows --> === S2 ===',
  '  --> WavesArmes',
  '  Applauds --> === S2 ===',
  '}',
  '',
  'partition Orchestra #CCCCEE {',
  '  WavesArmes --> Introduction',
  '  --> "Play music"',
  '}',
  '```'
].join('\n');
 
const allPluginsContent = [chartContent, codeContent, tableContent, umlContent].join('\n');
 


# In[ ]:


<!DOCTYPE html>
<html>
  <head lang="en">
    <meta charset="UTF-8" />
    <title>6. Editor with Chart Plugin</title>
    <link rel="stylesheet" href="./css/tuidoc-example-style.css" />
    <!-- Editor's Dependencies -->
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.48.4/codemirror.css"
    (>)
    <!-- Editor -->
    <link rel="stylesheet" href="../dist/cdn/toastui-editor.css" />
    <!-- Editor's Plugin -->
    <link rel="stylesheet" href="https://uicdn.toast.com/tui.chart/v3.7.0/tui-chart.css" />
  </head>
  <body>
    <div class="tui-doc-description">
      <strong
        >The example code can be slower than your environment because the code is transpiled by
        babel-standalone in runtime.</strong
      >
      <br />
      You can see the tutorial
      <a
        href="https://github.com/nhn/tui.editor/blob/master/apps/editor/docs/plugins.md"
        target="_blank"
        >here</a
      >.
    </div>
    <div class="code-html tui-doc-contents">
      <!-- Editor -->
      <h2>Editor</h2>
      <div id="editor"></div>
      <!-- Viewer Using Editor -->
      <h2>Viewer</h2>
      <div id="viewer"></div>
    </div>
    <!-- Added to check demo page in Internet Explorer -->
    <script src="https://unpkg.com/babel-standalone@6.26.0/babel.min.js"></script>
    <script src="./data/md-plugins.js"></script>
    <!-- Editor -->
    <script src="../dist/cdn/toastui-editor-all.js"></script>
    <!-- Editor's Plugin -->
    <script src="https://uicdn.toast.com/editor-plugin-chart/1.0.0/toastui-editor-plugin-chart.min.js"></script>
    <script type="text/babel" class="code-js">
      const { Editor } = toastui;
      const { chart } = Editor.plugin;
 
      const chartOptions = {
        minWidth: 100,
        maxWidth: 600,
        minHeight: 100,
        maxHeight: 300
      };
 
      const editor = new Editor({
        el: document.querySelector('#editor'),
        previewStyle: 'vertical',
        height: '500px',
        initialValue: chartContent,
        plugins: [chart, chartOptions]
      });
 
      const viewer = Editor.factory({
        el: document.querySelector('#viewer'),
        viewer: true,
        height: '500px',
        initialValue: chartContent,
        plugins: [[chart, chartOptions]]
      });
    </script>
  </body>
</html>
 


# In[ ]:


**()
 * @fileoverview configs file for bundling
 * @author NHN FE Development Lab <dl_javascript@nhn.com>
 */
const path = require('path');
const webpack = require('webpack');
const pkg = require('./package.json');
 
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const OptimizeCSSAssetsPlugin = require('optimize-css-assets-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const FileManagerPlugin = require('filemanager-webpack-plugin');
 
const ENTRY_EDITOR = './src/js/index.js';
const ENTRY_VIEWER = './src/js/indexViewer.js';
 
const isDevelopAll = process.argv.indexOf('--all') >= 0;
const isDevelopViewer = process.argv.indexOf('--viewer') >= 0;
const isProduction = process.argv.indexOf('--mode=production') >= 0;
const minify = process.argv.indexOf('--minify') >= 0;
 
const defaultConfigs = Array(isProduction ? 2 : 1)
  .fill(0)
  .map(() => {
    return {
      mode: isProduction ? 'production' : 'development',
      cache: false,
      output: {
        library: ['toastui', 'Editor'],
        libraryTarget: 'umd',
        libraryExport: 'default',
        path: path.resolve(__dirname, minify ? 'dist/cdn' : 'dist'),
        filename: `toastui-[name]${minify ? '.min' : ''}.js`
      },
      module: {
        rules: [
          {
            test: /\.js$/,
            exclude: /node_modules|dist|build/,
            loader: 'eslint-loader',
            enforce: 'pre',
            options: {
              configFile: './.eslintrc.js',
              failOnWarning: false,
              failOnError: false
            }
          },
          {
            test: /\.js$/,
            exclude: /node_modules|dist|build/,
            loader: 'babel-loader?cacheDirectory',
            options: {
              envName: isProduction ? 'production' : 'development',
              rootMode: 'upward'
            }
          },
          {
            test: /\.css$/,
            use: [MiniCssExtractPlugin.loader, 'css-loader']
          },
          {
            test: /\.png$/i,
            use: 'url-loader'
          }
        ]
      },
      plugins: [
        new MiniCssExtractPlugin({
          moduleFilename: ({ name }) =>
            `toastui-${name.replace('-all', '')}${minify ? '.min' : ''}.css`
        }),
        new webpack.BannerPlugin({
          banner: [
            pkg.name,
            `@version ${pkg.version} | ${new Date().toDateString()}`,
            `@author ${pkg.author}`,
            `@license ${pkg.license}`
          ].join('\n'),
          raw: false,
          entryOnly: true
        })
      ],
      externals: [
        {
          codemirror: {
            commonjs: 'codemirror',
            commonjs2: 'codemirror',
            amd: 'codemirror',
            root: ['CodeMirror']
          }
        }
      ],
      optimization: {
        minimize: false
      },
      performance: {
        hints: false
      }
    };
  });
 
function addFileManagerPlugin(config) {
  // When an entry option's value is set to a CSS file,
  (/, empty, JavaScript, files, are, created., (e.g., toastui-editor-only.js))
  (/, These, files, are, unnecessary,, so, use, the, FileManager, plugin, to, delete, them.)
  const options = minify
    get_ipython().run_line_magic('pinfo', '')
        {
          delete: [
            './dist/cdn/toastui-editor-only.min.js',
            './dist/cdn/toastui-editor-old.min.js',
            './dist/cdn/toastui-editor-viewer-old.min.js'
          ]
        }
      ]
    : [
        {
          delete: [
            './dist/toastui-editor-only.js',
            './dist/toastui-editor-old.js',
            './dist/toastui-editor-viewer-old.js'
          ]
        },
        { copy: [{ source: './dist/*.{js,css}', destination: './dist/cdn' }] }
      ];
 
  config.plugins.push(new FileManagerPlugin({ onEnd: options }));
}
 
function addMinifyPlugin(config) {
  config.optimization = {
    minimizer: [
      new TerserPlugin({
        cache: true,
        parallel: true,
        sourceMap: false,
        extractComments: false
      }),
      new OptimizeCSSAssetsPlugin()
    ]
  };
}
 
function addAnalyzerPlugin(config, type) {
  config.plugins.push(
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      reportFilename: `../../report/webpack/stats-${pkg.version}-${type}.html`
    })
  );
}
 
function setDevelopConfig(config) {
  if (isDevelopAll) {
    // check in examples
    config.entry = { 'editor-all': ENTRY_EDITOR };
    config.output.publicPath = 'dist/cdn';
    config.externals = [];
  } else if (isDevelopViewer) {
    // check in examples
    config.entry = { 'editor-viewer': ENTRY_VIEWER };
    config.output.publicPath = 'dist/cdn';
  } else {
    // check in demo
    config.module.rules = config.module.rules.slice(1);
    config.entry = { editor: ENTRY_EDITOR };
    config.output.publicPath = 'dist/';
  }
 
  config.devtool = 'inline-source-map';
  config.devServer = {
    inline: true,
    host: '0.0.0.0',
    port: 8080,
    disableHostCheck: true
  };
}
 
function setProductionConfig(config) {
  config.entry = {
    editor: ENTRY_EDITOR,
    'editor-viewer': ENTRY_VIEWER,
    'editor-only': './src/js/indexEditorOnlyStyle.js',
    // legacy styles
    'editor-old': './src/js/indexOldStyle.js',
    'editor-viewer-old': './src/css/old/contents.css'
  };
 
  addFileManagerPlugin(config);
 
  if (minify) {
    addMinifyPlugin(config);
    addAnalyzerPlugin(config, 'normal');
  }
}
 
function setProductionConfigForAll(config) {
  config.entry = { 'editor-all': ENTRY_EDITOR };
  config.output.path = path.resolve(__dirname, 'dist/cdn');
  config.externals = [];
 
  if (minify) {
    addMinifyPlugin(config);
    addAnalyzerPlugin(config, 'all');
  }
}
 
if (isProduction) {
  setProductionConfig(defaultConfigs[0]);
  setProductionConfigForAll(defaultConfigs[1]);
} else {
  setDevelopConfig(defaultConfigs[0]);
}
 
module.exports = defaultConfigs;
 


# In[ ]:


(/, Type, definitions, for, TOAST, UI, Editor, v2.1.0)
(/, TypeScript, Version:, 3.2.2)
 
(//, <reference, types="codemirror", />)
 
declare namespace toastui {
  type SquireExt = any;
  type HandlerFunc = (...args: any[]) => void;
  type ReplacerFunc = (inputString: string) => string;
  type CodeMirrorType = CodeMirror.EditorFromTextArea;
  type CommandManagerExecFunc = (name: string, ...args: any[]) => any;
  type PopupTableUtils = LayerPopup;
  type AddImageBlobHook = (fileOrBlob: File | Blob, callback: Function, source: string) => void;
  type Plugin = (editor: Editor | Viewer, options: any) => void;
  type PreviewStyle = 'tab' | 'vertical';
  type CustomHTMLSanitizer = (content: string) => string | DocumentFragment;
  type LinkAttribute = Partial<{
    rel: string;
    target: string;
    contenteditable: boolean | 'true' | 'false';
    hreflang: string;
    type: string;
  }>;
  type AutolinkParser = (
    content: string
  ) => {
    url: string;
    text: string;
    range: [number, number];
  }[];
  type ExtendedAutolinks = boolean | AutolinkParser;
  type Sanitizer = (content: string) => string | DocumentFragment;
 
  // @TODO: change toastMark type definition to @toast-ui/toastmark type file through importing
  // Toastmark custom renderer type
  type BlockNodeType =
    | 'document'
    | 'list'
    | 'blockQuote'
    | 'item'
    | 'heading'
    | 'thematicBreak'
    | 'paragraph'
    | 'codeBlock'
    | 'htmlBlock'
    | 'table'
    | 'tableHead'
    | 'tableBody'
    | 'tableRow'
    | 'tableCell'
    | 'tableDelimRow'
    | 'tableDelimCell'
    | 'refDef';
 
  type InlineNodeType =
    | 'code'
    | 'text'
    | 'emph'
    | 'strong'
    | 'strike'
    | 'link'
    | 'image'
    | 'htmlInline'
    | 'linebreak'
    | 'softbreak';
 
  type NodeType = BlockNodeType | InlineNodeType;
  type SourcePos = [[number, number], [number, number]];
 
  interface NodeWalker {
    current: MdNode | null;
    root: MdNode;
    entering: boolean;
 
    next(): { entering: boolean; node: MdNode } | null;
    resumeAt(node: MdNode, entering: boolean): void;
  }
 
  interface MdNode {
    type: NodeType;
    id: number;
    parent: MdNode | null;
    prev: MdNode | null;
    next: MdNode | null;
    sourcepos?: SourcePos;
    firstChild: MdNode | null;
    lastChild: MdNode | null;
    literal: string | null;
 
    isContainer(): boolean;
    unlink(): void;
    replaceWith(node: MdNode): void;
    insertAfter(node: MdNode): void;
    insertBefore(node: MdNode): void;
    appendChild(child: MdNode): void;
    prependChild(child: MdNode): void;
    walker(): NodeWalker;
  }
 
  interface TagToken {
    tagName: string;
    outerNewLine?: boolean;
    innerNewLine?: boolean;
  }
 
  interface OpenTagToken extends TagToken {
    type: 'openTag';
    classNames?: string[];
    attributes?: Record<string, string>;
    selfClose?: boolean;
  }
 
  interface CloseTagToken extends TagToken {
    type: 'closeTag';
  }
 
  interface TextToken {
    type: 'text';
    content: string;
  }
 
  interface RawHTMLToken {
    type: 'html';
    content: string;
    outerNewLine?: boolean;
  }
 
  type HTMLToken = OpenTagToken | CloseTagToken | TextToken | RawHTMLToken;
 
  interface ContextOptions {
    gfm: boolean;
    softbreak: string;
    nodeId: boolean;
    tagFilter: boolean;
    convertors?: CustomHTMLRendererMap;
  }
 
  interface Context {
    entering: boolean;
    leaf: boolean;
    options: Omit<ContextOptions, 'gfm' | 'convertors'>;
    getChildrenText: (node: MdNode) => string;
    skipChildren: () => void;
    origin?: () => ReturnType<CustomHTMLRenderer>;
  }
 
  export type CustomHTMLRenderer = (node: MdNode, context: Context) => HTMLToken | HTMLToken[] | null;
 
  type CustomHTMLRendererMap = Partial<Record<NodeType, CustomHTMLRenderer>>;
  (/, Toastmark, custom, renderer, type, end)
  interface SelectionRange {
    from: {
      row: number;
      cell: number;
    };
    to: {
      row: number;
      cell: number;
    };
  }
 
  interface ToolbarState {
    strong: boolean;
    emph: boolean;
    strike: boolean;
    code: boolean;
    codeBlock: boolean;
    blockQuote: boolean;
    table: boolean;
    heading: boolean;
    list: boolean;
    orderedList: boolean;
    taskList: boolean;
  }
 
  type WysiwygToolbarState = ToolbarState & {
    source: 'wysiwyg';
  };
 
  type MarkdownToolbarState = ToolbarState & {
    thematicBreak: boolean;
    source: 'markdown';
  };
 
  type SourceType = 'wysiwyg' | 'markdown';
 
  interface EventMap {
    load?: (param: Editor) => void;
    change?: (param: { source: SourceType | 'viewer'; data: MouseEvent }) => void;
    stateChange?: (param: MarkdownToolbarState | WysiwygToolbarState) => void;
    focus?: (param: { source: SourceType }) => void;
    blur?: (param: { source: SourceType }) => void;
  }
 
  interface ViewerHookMap {
    previewBeforeHook?: (html: string) => void | string;
  }
 
  type EditorHookMap = ViewerHookMap & {
    addImageBlobHook?: (
      blob: Blob | File,
      callback: (url: string, altText: string) => void
    ) => void;
  };
 
  interface ToMarkOptions {
    gfm?: boolean;
    renderer?: any;
  }
 
  export interface Convertor {
    initHtmlSanitizer(sanitizer: Sanitizer): void;
    toHTML(makrdown: string): string;
    toHTMLWithCodeHighlight(markdown: string): string;
    toMarkdown(html: string, toMarkdownOptions: ToMarkOptions): string;
  }
 
  export interface ConvertorClass {
    new (em: EventManager, options: ConvertorOptions): Convertor;
  }
 
  export interface ConvertorOptions {
    linkAttribute: LinkAttribute;
    customHTMLRenderer: CustomHTMLRenderer;
    extendedAutolinks: boolean | AutolinkParser;
    referenceDefinition: boolean;
  }
 
  export interface EditorOptions {
    el: HTMLElement;
    height?: string;
    minHeight?: string;
    initialValue?: string;
    previewStyle?: PreviewStyle;
    initialEditType?: string;
    events?: EventMap;
    hooks?: EditorHookMap;
    language?: string;
    useCommandShortcut?: boolean;
    useDefaultHTMLSanitizer?: boolean;
    usageStatistics?: boolean;
    toolbarItems?: (string | ToolbarButton)[];
    hideModeSwitch?: boolean;
    plugins?: Plugin[];
    extendedAutolinks?: ExtendedAutolinks;
    customConvertor?: ConvertorClass;
    placeholder?: string;
    linkAttribute?: LinkAttribute;
    customHTMLRenderer?: CustomHTMLRenderer;
    referenceDefinition?: boolean;
    customHTMLSanitizer?: CustomHTMLSanitizer;
    previewHighlight?: boolean;
  }
 
  export interface ViewerOptions {
    el: HTMLElement;
    initialValue?: string;
    events?: EventMap;
    hooks?: ViewerHookMap;
    plugins?: Plugin[];
    useDefaultHTMLSanitizer?: boolean;
    extendedAutolinks?: ExtendedAutolinks;
    customConvertor?: ConvertorClass;
    linkAttribute?: LinkAttribute;
    customHTMLRenderer?: CustomHTMLRenderer;
    referenceDefinition?: boolean;
    customHTMLSanitizer?: CustomHTMLSanitizer;
  }
 
  interface MarkdownEditorOptions {
    height?: string;
  }
 
  interface WysiwygEditorOptions {
    useDefaultHTMLSanitizer?: boolean;
    linkAttribute?: LinkAttribute;
  }
 
  interface LanguageData {
    [propType: string]: string;
  }
 
  interface ToolbarButton {
    type: string;
    options: ButtonOptions;
  }
 
  interface ButtonOptions {
    el?: HTMLElement;
    className?: string;
    command?: string;
    event?: string;
    text?: string;
    tooltip?: string;
    style?: string;
    state?: string;
  }
 
  class UIController {
    public tagName: string;
 
    public className: string;
 
    public el: HTMLElement;
 
    public on(aType: string | object, aFn: (...args: any[]) => void): void;
 
    public off(type: string, fn: (...args: any[]) => void): void;
 
    public remove(): void;
 
    public trigger(eventTypeEvent: string, eventData?: any): void;
 
    public destroy(): void;
  }
 
  class ToolbarItem extends UIController {
    public static name: string;
 
    public static className: string;
 
    public getName(): string;
  }
 
  interface CommandType {
    MD: 0;
    WW: 1;
    GB: 2;
  }
 
  interface CommandProps {
    name: string;
    type: number;
  }
 
  class Command {
    public static TYPE: CommandType;
 
    public static factory(typeStr: string, props: CommandProps): Command;
 
    constructor(name: string, type: number, keyMap?: string[]);
 
    public getName(): string;
 
    public getType(): number;
 
    public isGlobalType(): boolean;
 
    public isMDType(): boolean;
 
    public isWWType(): boolean;
 
    public setKeyMap(win: string, mac: string): void;
  }
 
  interface LayerPopupOptions {
    openerCssQuery?: string[];
    closerCssQuery?: string[];
    el: HTMLElement;
    content?: HTMLElement | string;
    textContent?: string;
    title: string;
    header?: boolean;
    target?: HTMLElement;
    modal: boolean;
    headerButtons?: string;
  }
 
  interface LayerPopup extends UIController {
    setContent(content: HTMLElement): void;
    setTitle(title: string): void;
    getTitleElement(): HTMLElement;
    hide(): void;
    show(): void;
    isShow(): boolean;
    remove(): void;
    setFitToWindow(fit: boolean): void;
    isFitToWindow(): boolean;
    toggleFitToWindow(): boolean;
  }
 
  interface ModeSwitchType {
    MARKDOWN: 'markdown';
    WYSIWYG: 'wysiwyg';
  }
 
  interface ModeSwitch extends UIController {
    TYPE: ModeSwitchType;
    isShown(): boolean;
    show(): void;
    hide(): void;
  }
 
  class Toolbar extends UIController {
    public disableAllButton(): void;
 
    public enableAllButton(): void;
 
    public getItems(): ToolbarItem[];
 
    public getItem(index: number): ToolbarItem;
 
    public setItems(items: ToolbarItem[]): void;
 
    public addItem(item: ToolbarItem | ToolbarButton | string): void;
 
    public insertItem(index: number, item: ToolbarItem | ToolbarButton | string): void;
 
    public indexOfItem(item: ToolbarItem): number;
 
    public removeItem(item: ToolbarItem | number, destroy?: boolean): ToolbarItem | undefined;
 
    public removeAllItems(): void;
  }
 
  interface UI {
    createPopup(options: LayerPopupOptions): LayerPopup;
    getEditorHeight(): number;
    getEditorSectionHeight(): number;
    getModeSwitch(): ModeSwitch;
    getPopupTableUtils(): PopupTableUtils;
    getToolbar(): Toolbar;
    hide(): void;
    remove(): void;
    setToolbar(toolbar: Toolbar): void;
    show(): void;
  }
 
  interface CommandManagerOptions {
    useCommandShortcut?: boolean;
  }
 
  interface CommandPropsOptions {
    name: string;
    keyMap?: string[];
    exec?: CommandManagerExecFunc;
  }
 
  class CommandManager {
    public static command(type: string, props: CommandPropsOptions): Command;
 
    constructor(base: Editor, options?: CommandManagerOptions);
 
    public addCommand(command: Command): Command;
 
    public exec(name: string, ...args: any[]): any;
  }
 
  class CodeBlockManager {
    public createCodeBlockHtml(language: string, codeText: string): string;
 
    public getReplacer(language: string): ReplacerFunc;
 
    public setReplacer(language: string, replacer: ReplacerFunc): void;
  }
 
  interface RangeType {
    start: {
      line: number;
      ch: number;
    };
    end: {
      line: number;
      ch: number;
    };
  }
 
  interface MdTextObject {
    setRange(range: RangeType): void;
    setEndBeforeRange(range: RangeType): void;
    expandStartOffset(): void;
    expandEndOffset(): void;
    getTextContent(): RangeType;
    replaceContent(content: string): void;
    deleteContent(): void;
    peekStartBeforeOffset(offset: number): RangeType;
  }
 
  interface WwTextObject {
    deleteContent(): void;
    expandEndOffset(): void;
    expandStartOffset(): void;
    getTextContent(): string;
    peekStartBeforeOffset(offset: number): string;
    replaceContent(content: string): void;
    setEndBeforeRange(range: Range): void;
    setRange(range: Range): void;
  }
 
  interface FindOffsetNodeInfo {
    container: Node;
    offsetInContainer: number;
    offset: number;
  }
 
  interface NodeInfo {
    id?: string;
    tagName: string;
    className?: string;
  }
 
  class WwCodeBlockManager {
    constructor(wwe: WysiwygEditor);
 
    public destroy(): void;
 
    public convertNodesToText(nodes: Node[]): string;
 
    public isInCodeBlock(range: Range): boolean;
 
    public prepareToPasteOnCodeblock(nodes: Node[]): DocumentFragment;
 
    public modifyCodeBlockForWysiwyg(node: HTMLElement): void;
  }
 
  class WwTableManager {
    constructor(wwe: WysiwygEditor);
 
    public destroy(): void;
 
    public getTableIDClassName(): string;
 
    public isInTable(range: Range): boolean;
 
    public isNonTextDeleting(range: Range): boolean;
 
    public isTableOrSubTableElement(pastingNodeName: string): boolean;
 
    public pasteClipboardData(clipboardTable: Node): boolean;
 
    public prepareToTableCellStuffing(
      trs: HTMLElement
    ): { maximumCellLength: number; needTableCellStuffingAid: boolean };
 
    public resetLastCellNode(): void;
 
    public setLastCellNode(node: HTMLElement): void;
 
    public tableCellAppendAidForTableElement(node: HTMLElement): void;
 
    public updateTableHtmlOfClipboardIfNeed(clipboardContainer: HTMLElement): void;
 
    public wrapDanglingTableCellsIntoTrIfNeed(container: HTMLElement): HTMLElement | null;
 
    public wrapTheadAndTbodyIntoTableIfNeed(container: HTMLElement): HTMLElement | null;
 
    public wrapTrsIntoTbodyIfNeed(container: HTMLElement): HTMLElement | null;
  }
 
  class WwTableSelectionManager {
    constructor(wwe: WysiwygEditor);
 
    public createRangeBySelectedCells(): void;
 
    public destroy(): void;
 
    public getSelectedCells(): HTMLElement;
 
    public getSelectionRangeFromTable(
      selectionStart: HTMLElement,
      selectionEnd: HTMLElement
    ): SelectionRange;
 
    public highlightTableCellsBy(selectionStart: HTMLElement, selectionEnd: HTMLElement): void;
 
    public removeClassAttrbuteFromAllCellsIfNeed(): void;
 
    public setTableSelectionTimerIfNeed(selectionStart: HTMLElement): void;
 
    public styleToSelectedCells(onStyle: SquireExt, options?: object): void;
  }
 
  (/, @TODO:, change, toastMark, type, definition, to, @toast-ui/toastmark, type, file, through, importing)
  class MarkdownEditor {
    static factory(
      el: HTMLElement,
      eventManager: EventManager,
      toastMark: any,
      options: MarkdownEditorOptions
    ): MarkdownEditor;
 
    constructor(
      el: HTMLElement,
      eventManager: EventManager,
      toastMark: any,
      options: MarkdownEditorOptions
    );
 
    public getTextObject(range: Range | RangeType): MdTextObject;
 
    public setValue(markdown: string, cursorToEnd?: boolean): void;
 
    public resetState(): void;
 
    public getMdDocument(): any;
  }
 
  class WysiwygEditor {
    static factory(
      el: HTMLElement,
      eventManager: EventManager,
      options: WysiwygEditorOptions
    ): WysiwygEditor;
 
    constructor(el: HTMLElement, eventManager: EventManager, options: WysiwygEditorOptions);
 
    public addKeyEventHandler(keyMap: string | string[], handler: HandlerFunc): void;
 
    public addWidget(range: Range, node: Node, style: string, offset?: number): void;
 
    public blur(): void;
 
    public breakToNewDefaultBlock(range: Range, where?: string): void;
 
    public changeBlockFormatTo(targetTagName: string): void;
 
    public findTextNodeFilter(): boolean;
 
    public fixIMERange(): void;
 
    public focus(): void;
 
    public getEditor(): SquireExt;
 
    public getIMERange(): Range;
 
    public getRange(): Range;
 
    public getTextObject(range: Range): WwTextObject;
 
    public getValue(): string;
 
    public hasFormatWithRx(rx: RegExp): boolean;
 
    public init(useDefaultHTMLSanitizer: boolean): void;
 
    public insertText(text: string): void;
 
    public makeEmptyBlockCurrentSelection(): void;
 
    public moveCursorToEnd(): void;
 
    public moveCursorToStart(): void;
 
    public postProcessForChange(): void;
 
    public readySilentChange(): void;
 
    public remove(): void;
 
    public removeKeyEventHandler(keyMap: string, handler: HandlerFunc): void;
 
    public replaceContentText(container: Node, from: string, to: string): void;
 
    public replaceRelativeOffset(content: string, offset: number, overwriteLength: number): void;
 
    public replaceSelection(content: string, range: Range): void;
 
    public reset(): void;
 
    public restoreSavedSelection(): void;
 
    public saveSelection(range: Range): void;
 
    public scrollTop(value: number): boolean;
 
    public setHeight(height: number | string): void;
 
    public setPlaceholder(placeholder: string): void;
 
    public setMinHeight(minHeight: number): void;
 
    public setRange(range: Range): void;
 
    public getLinkAttribute(): LinkAttribute;
 
    public setSelectionByContainerAndOffset(
      startContainer: Node,
      startOffset: number,
      endContainer: Node,
      endOffset: number
    ): Range;
 
    public setValue(html: string, cursorToEnd?: boolean): void;
 
    public unwrapBlockTag(condition?: (tagName: string) => boolean): void;
 
    public getBody(): HTMLElement;
 
    public scrollIntoCursor(): void;
 
    public isInTable(range: Range): boolean;
  }
 
  class EventManager {
    public addEventType(type: string): void;
 
    public emit(eventName: string): any[];
 
    public emitReduce(eventName: string, sourceText: string): string;
 
    public listen(typeStr: string, handler: HandlerFunc): void;
 
    public removeEventHandler(typeStr: string, handler?: HandlerFunc): void;
  }
 
  export class Editor {
    public static codeBlockManager: CodeBlockManager;
 
    public static CommandManager: CommandManager;
 
    public static isViewer: boolean;
 
    public static WwCodeBlockManager: WwCodeBlockManager;
 
    public static WwTableManager: WwTableManager;
 
    public static WwTableSelectionManager: WwTableSelectionManager;
 
    public static factory(options: EditorOptions): Editor | Viewer;
 
    public static getInstances(): Editor[];
 
    public static setLanguage(code: string, data: LanguageData): void;
 
    constructor(options: EditorOptions);
 
    public addHook(type: string, handler: HandlerFunc): void;
 
    public addWidget(selection: Range, node: Node, style: string, offset?: number): void;
 
    public afterAddedCommand(): void;
 
    public blur(): void;
 
    public changeMode(mode: string, isWithoutFocus?: boolean): void;
 
    public changePreviewStyle(style: PreviewStyle): void;
 
    public exec(name: string, ...args: any[]): void;
 
    public focus(): void;
 
    public getCodeMirror(): CodeMirrorType;
 
    public getCurrentModeEditor(): MarkdownEditor | WysiwygEditor;
 
    public getCurrentPreviewStyle(): PreviewStyle;
 
    public getHtml(): string;
 
    public getMarkdown(): string;
 
    public getRange(): Range | RangeType;
 
    public getSelectedText(): string;
 
    public getSquire(): SquireExt;
 
    public getTextObject(range: Range | RangeType): MdTextObject | WwTextObject;
 
    public getUI(): UI;
 
    public getValue(): string;
 
    public height(height: string): string;
 
    public hide(): void;
 
    public insertText(text: string): void;
 
    public isMarkdownMode(): boolean;
 
    public isViewer(): boolean;
 
    public isWysiwygMode(): boolean;
 
    public minHeight(minHeight: string): string;
 
    public moveCursorToEnd(): void;
 
    public moveCursorToStart(): void;
 
    public off(type: string): void;
 
    public on(type: string, handler: HandlerFunc): void;
 
    public remove(): void;
 
    public removeHook(type: string): void;
 
    public reset(): void;
 
    public scrollTop(value: number): number;
 
    public setHtml(html: string, cursorToEnd?: boolean): void;
 
    public setMarkdown(markdown: string, cursorToEnd?: boolean): void;
 
    public setUI(UI: UI): void;
 
    public setValue(value: string, cursorToEnd?: boolean): void;
 
    public show(): void;
 
    public setCodeBlockLanguages(languages?: string[]): void;
  }
 
  export class Viewer {
    public static isViewer: boolean;
 
    public static codeBlockManager: CodeBlockManager;
 
    public static WwCodeBlockManager: null;
 
    public static WwTableManager: null;
 
    public static WwTableSelectionManager: null;
 
    constructor(options: ViewerOptions);
 
    public addHook(type: string, handler: HandlerFunc): void;
 
    public isMarkdownMode(): boolean;
 
    public isViewer(): boolean;
 
    public isWysiwygMode(): boolean;
 
    public off(type: string): void;
 
    public on(type: string, handler: HandlerFunc): void;
 
    public remove(): void;
 
    public setMarkdown(markdown: string): void;
 
    public setValue(markdown: string): void;
 
    public setCodeBlockLanguages(languages?: string[]): void;
  }
}
 
declare module '@toast-ui/editor' {
  export type EditorOptions = toastui.EditorOptions;
  export type CustomConvertor = toastui.ConvertorClass;
  export type EventMap = toastui.EventMap;
  export type EditorHookMap = toastui.EditorHookMap;
  export type CustomHTMLRenderer = toastui.CustomHTMLRenderer;
  export type ExtendedAutolinks = toastui.ExtendedAutolinks;
  export type LinkAttribute = toastui.LinkAttribute;
  export default toastui.Editor;
}
 
declare module '@toast-ui/editor/dist/toastui-editor-viewer' {
  export type ViewerOptions = toastui.ViewerOptions;
  export type CustomConvertor = toastui.ConvertorClass;
  export type EventMap = toastui.EventMap;
  export type ViewerHookMap = toastui.ViewerHookMap;
  export type CustomHTMLRenderer = toastui.CustomHTMLRenderer;
  export type ExtendedAutolinks = toastui.ExtendedAutolinks;
  export type LinkAttribute = toastui.LinkAttribute;
  export default toastui.Viewer;
}
 


# In[ ]:


{
  "name": "@toast-ui/editor",
  "version": "2.1.0",
  "description": "GFM  Markdown Wysiwyg Editor - Productive and Extensible",
  "keywords": [
    "nhn",
    "toast",
    "toastui",
    "toast-ui",
    "markdown",
    "wysiwyg",
    "editor",
    "preview",
    "gfm"
  ],
  "main": "dist/toastui-editor.js",
  "files": [
    "dist/*.js",
    "dist/*.css",
    "dist/i18n",
    "index.d.ts"
  ],
  "author": "NHN FE Development Lab <dl_javascript@nhn.com>",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/nhn/tui.editor.git",
    "directory": "apps/editor"
  },
  "bugs": {
    "url": "https://github.com/nhn/tui.editor/issues"
  },
  "homepage": "https://ui.toast.com",
  "browserslist": "last 2 versions, not ie <= 10",
  "scripts": {
    "lint": "eslint .",
    "test": "karma start --no-single-run",
    "test:ne": "cross-env KARMA_SERVER=ne karma start",
    "test:types": "tsc --project test/types",
    "e2e": "testcafe chrome 'test/e2e/**/*.spec.js'",
    "e2e:sl": "testcafe \"saucelabs:Chrome@65.0:Windows 10,saucelabs:Firefox@59.0:Windows 10,saucelabs:Safari@10.0:OS X 10.11,saucelabs:Internet Explorer@11.103:Windows 10,saucelabs:MicrosoftEdge@16.16299:Windows 10\" 'test/e2e/**/*.spec.js'",
    "serve": "webpack-dev-server",
    "serve:viewer": "webpack-dev-server --viewer",
    "serve:all": "webpack-dev-server --all",
    "build:i18n": "cross-env webpack --config scripts/webpack.config.i18n.js && webpack --config scripts/webpack.config.i18n.js --minify",
    "build:prod": "cross-env webpack --mode=production && webpack --mode=production --minify && node tsBannerGenerator.js",
    "build": "npm run build:i18n && npm run build:prod",
    "note": "tui-note --tag=$(git describe --tags)",
    "tslint": "tslint index.d.ts",
    "doc:serve": "tuidoc --serv",
    "doc": "tuidoc"
  },
  "devDependencies": {
    "@babel/core": "^7.8.3",
    "@babel/plugin-proposal-class-properties": "^7.8.3",
    "@babel/preset-env": "^7.8.3",
    "@toast-ui/release-notes": "^2.0.1",
    "@toast-ui/squire": "file:../../libs/squire",
    "@toast-ui/to-mark": "file:../../libs/to-mark",
    "@toast-ui/toastmark": "file:../../libs/toastmark",
    "babel-eslint": "^10.0.3",
    "babel-loader": "^8.0.6",
    "babel-plugin-istanbul": "^6.0.0",
    "cross-env": "^6.0.3",
    "css-loader": "^3.4.2",
    "eslint": "^6.8.0",
    "eslint-config-prettier": "^6.9.0",
    "eslint-config-tui": "^3.0.0",
    "eslint-loader": "^3.0.3",
    "eslint-plugin-prettier": "^3.1.2",
    "filemanager-webpack-plugin": "^2.0.5",
    "istanbul-instrumenter-loader": "^3.0.1",
    "jasmine-core": "^2.99.1",
    "jquery": "^3.4.1",
    "karma": "^4.4.1",
    "karma-chrome-launcher": "^3.1.0",
    "karma-coverage-istanbul-reporter": "^2.1.1",
    "karma-jasmine": "^1.1.2",
    "karma-jasmine-ajax": "^0.1.13",
    "karma-jasmine-jquery": "^0.1.1",
    "karma-sourcemap-loader": "^0.3.7",
    "karma-webdriver-launcher": "github:nhn/karma-webdriver-launcher#v1.2.0",
    "karma-webpack": "^4.0.2",
    "mini-css-extract-plugin": "^0.9.0",
    "optimize-css-assets-webpack-plugin": "^5.0.3",
    "prettier": "^1.19.1",
    "resize-observer-polyfill": "^1.5.1",
    "terser-webpack-plugin": "^2.2.1",
    "testcafe": "^0.23.3",
    "testcafe-browser-provider-saucelabs": "^1.3.0",
    "tslint": "^5.12.0",
    "tui-code-snippet": "^2.3.1",
    "typescript": "^3.2.2",
    "url-loader": "^3.0.0",
    "webpack": "^4.40.2",
    "webpack-bundle-analyzer": "^3.6.0",
    "webpack-cli": "^3.3.9",
    "webpack-dev-server": "^3.1.11",
    "webpack-glob-entry": "^2.1.1"
  },
  "dependencies": {
    "@types/codemirror": "0.0.71",
    "codemirror": "^5.48.4"
  }
}
 


# In[ ]:


**()
 * @fileoverview Configs for i18n bundle file
 * @author NHN FE Development Lab <dl_javascript@nhn.com>
 */
const path = require('path');
const webpack = require('webpack');
const entry = require('webpack-glob-entry');
const pkg = require('../package.json');
 
const TerserPlugin = require('terser-webpack-plugin');
const FileManagerPlugin = require('filemanager-webpack-plugin');
 
function getOptimizationConfig(minify) {
  const minimizer = [];
 
  if (minify) {
    minimizer.push(
      new TerserPlugin({
        cache: true,
        parallel: true,
        sourceMap: false,
        extractComments: false
      })
    );
  }
 
  return { minimizer };
}
 
function getEntries() {
  const entries = entry('./src/js/i18n/*.js');
 
  delete entries['en-us'];
 
  return entries;
}
 
module.exports = (env, argv) => {
  const minify = !!argv.minify;
 
  return {
    mode: 'production',
    entry: getEntries(),
    output: {
      libraryTarget: 'umd',
      path: path.resolve(__dirname, minify ? '../dist/cdn/i18n' : '../dist/i18n'),
      filename: `[name]${minify ? '.min' : ''}.js`
    },
    externals: [
      {
        '../editor': {
          commonjs: '@toast-ui/editor',
          commonjs2: '@toast-ui/editor',
          amd: '@toast-ui/editor',
          root: ['toastui', 'Editor']
        }
      }
    ],
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          loader: 'eslint-loader',
          enforce: 'pre',
          options: {
            failOnError: true
          }
        },
        {
          test: /\.js$/,
          exclude: /node_modules|dist/,
          loader: 'babel-loader?cacheDirectory',
          options: {
            rootMode: 'upward'
          }
        }
      ]
    },
    plugins: [
      new webpack.BannerPlugin(
        [
          'TOAST UI Editor : i18n',
          `@version ${pkg.version}`,
          `@author ${pkg.author}`,
          `@license ${pkg.license}`
        ].join('\n')
      ),
      new FileManagerPlugin({
        onEnd: {
          copy: [{ source: './dist/i18n/*.js', destination: './dist/cdn/i18n' }]
        }
      })
    ],
    optimization: getOptimizationConfig(minify)
  };
};
 

