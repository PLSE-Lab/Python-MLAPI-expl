
import os
print(os.listdir("../input/nvidiaapex/repository/NVIDIA-apex-39e153a"))
#print(os.listdir("../input/glove-global-vectors-for-word-representation"))
#print(os.listdir("../input/jigsaw-unintended-bias-in-toxicity-classification"))
#print(os.listdir("../input/fasttext-crawl-300d-2m"))

# Any results you write to the current directory are saved as output.
['docs', '.gitignore', 'LICENSE', 'csrc', 'setup.py', 'apex', 'README.md', '.nojekyll', 'tests', 'examples']
# Installing Nvidia Apex
! pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/repository/NVIDIA-apex-39e153a
/opt/conda/lib/python3.6/site-packages/pip/_internal/commands/install.py:207: UserWarning: Disabling all use of wheels due to the use of --build-options / --global-options / --install-options.
  cmdoptions.check_install_build_global(options)
Created temporary directory: /tmp/pip-ephem-wheel-cache-nq_k_aq0
Created temporary directory: /tmp/pip-req-tracker-5n8zf4po
Created requirements tracker '/tmp/pip-req-tracker-5n8zf4po'
Created temporary directory: /tmp/pip-install-u_h0w8n3
Processing /kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a
  Created temporary directory: /tmp/pip-req-build-wajk9v9c
  Added file:///kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a to build tracker '/tmp/pip-req-tracker-5n8zf4po'
    Running setup.py (path:/tmp/pip-req-build-wajk9v9c/setup.py) egg_info for package from file:///kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a
    Running command python setup.py egg_info
    torch.__version__  =  1.0.1.post2
    running egg_info
    creating pip-egg-info/apex.egg-info
    writing pip-egg-info/apex.egg-info/PKG-INFO
    writing dependency_links to pip-egg-info/apex.egg-info/dependency_links.txt
    writing top-level names to pip-egg-info/apex.egg-info/top_level.txt
    writing manifest file 'pip-egg-info/apex.egg-info/SOURCES.txt'
    reading manifest file 'pip-egg-info/apex.egg-info/SOURCES.txt'
    writing manifest file 'pip-egg-info/apex.egg-info/SOURCES.txt'
  Source in /tmp/pip-req-build-wajk9v9c has version 0.1, which satisfies requirement apex==0.1 from file:///kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a
  Removed apex==0.1 from file:///kaggle/input/nvidiaapex/repository/NVIDIA-apex-39e153a from build tracker '/tmp/pip-req-tracker-5n8zf4po'
Installing collected packages: apex
  Created temporary directory: /tmp/pip-record-sz37cy8d
  Running setup.py install for apex ...     Running command /opt/conda/bin/python -u -c "import setuptools, tokenize;__file__='/tmp/pip-req-build-wajk9v9c/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" --cpp_ext --cuda_ext install --record /tmp/pip-record-sz37cy8d/install-record.txt --single-version-externally-managed --compile
    torch.__version__  =  1.0.1.post2

    Compiling cuda extensions with
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2018 NVIDIA Corporation
    Built on Sat_Aug_25_21:08:01_CDT_2018
    Cuda compilation tools, release 10.0, V10.0.130
    from /usr/local/cuda/bin

    Pytorch binaries were compiled with Cuda 10.0.130

    running install
    running build
    running build_py
    creating build
    creating build/lib.linux-x86_64-3.6
    creating build/lib.linux-x86_64-3.6/apex
    copying apex/__init__.py -> build/lib.linux-x86_64-3.6/apex
    creating build/lib.linux-x86_64-3.6/apex/reparameterization
    copying apex/reparameterization/__init__.py -> build/lib.linux-x86_64-3.6/apex/reparameterization
    copying apex/reparameterization/reparameterization.py -> build/lib.linux-x86_64-3.6/apex/reparameterization
    copying apex/reparameterization/weight_norm.py -> build/lib.linux-x86_64-3.6/apex/reparameterization
    creating build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/sync_batchnorm_kernel.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/__init__.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/LARC.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/sync_batchnorm.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/distributed.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/optimized_sync_batchnorm.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/multiproc.py -> build/lib.linux-x86_64-3.6/apex/parallel
    copying apex/parallel/optimized_sync_batchnorm_kernel.py -> build/lib.linux-x86_64-3.6/apex/parallel
    creating build/lib.linux-x86_64-3.6/apex/normalization
    copying apex/normalization/fused_layer_norm.py -> build/lib.linux-x86_64-3.6/apex/normalization
    copying apex/normalization/__init__.py -> build/lib.linux-x86_64-3.6/apex/normalization
    creating build/lib.linux-x86_64-3.6/apex/fp16_utils
    copying apex/fp16_utils/__init__.py -> build/lib.linux-x86_64-3.6/apex/fp16_utils
    copying apex/fp16_utils/fp16util.py -> build/lib.linux-x86_64-3.6/apex/fp16_utils
    copying apex/fp16_utils/loss_scaler.py -> build/lib.linux-x86_64-3.6/apex/fp16_utils
    copying apex/fp16_utils/fp16_optimizer.py -> build/lib.linux-x86_64-3.6/apex/fp16_utils
    creating build/lib.linux-x86_64-3.6/apex/optimizers
    copying apex/optimizers/__init__.py -> build/lib.linux-x86_64-3.6/apex/optimizers
    copying apex/optimizers/fp16_optimizer.py -> build/lib.linux-x86_64-3.6/apex/optimizers
    copying apex/optimizers/fused_adam.py -> build/lib.linux-x86_64-3.6/apex/optimizers
    creating build/lib.linux-x86_64-3.6/apex/RNN
    copying apex/RNN/models.py -> build/lib.linux-x86_64-3.6/apex/RNN
    copying apex/RNN/__init__.py -> build/lib.linux-x86_64-3.6/apex/RNN
    copying apex/RNN/RNNBackend.py -> build/lib.linux-x86_64-3.6/apex/RNN
    copying apex/RNN/cells.py -> build/lib.linux-x86_64-3.6/apex/RNN
    creating build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/compat.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/__init__.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/__version__.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/wrap.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/_initialize.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/handle.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/frontend.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/utils.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/opt.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/_process_optimizer.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/_amp_state.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/amp.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/rnn_compat.py -> build/lib.linux-x86_64-3.6/apex/amp
    copying apex/amp/scaler.py -> build/lib.linux-x86_64-3.6/apex/amp
    creating build/lib.linux-x86_64-3.6/apex/multi_tensor_apply
    copying apex/multi_tensor_apply/multi_tensor_apply.py -> build/lib.linux-x86_64-3.6/apex/multi_tensor_apply
    copying apex/multi_tensor_apply/__init__.py -> build/lib.linux-x86_64-3.6/apex/multi_tensor_apply
    creating build/lib.linux-x86_64-3.6/apex/amp/lists
    copying apex/amp/lists/torch_overrides.py -> build/lib.linux-x86_64-3.6/apex/amp/lists
    copying apex/amp/lists/tensor_overrides.py -> build/lib.linux-x86_64-3.6/apex/amp/lists
    copying apex/amp/lists/__init__.py -> build/lib.linux-x86_64-3.6/apex/amp/lists
    copying apex/amp/lists/functional_overrides.py -> build/lib.linux-x86_64-3.6/apex/amp/lists
    running build_ext
    building 'apex_C' extension
    creating build/temp.linux-x86_64-3.6
    creating build/temp.linux-x86_64-3.6/csrc
    gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/opt/conda/include/python3.6m -c csrc/flatten_unflatten.cpp -o build/temp.linux-x86_64-3.6/csrc/flatten_unflatten.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=apex_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
    g++ -pthread -shared -B /opt/conda/compiler_compat -L/opt/conda/lib -Wl,-rpath=/opt/conda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/csrc/flatten_unflatten.o -o build/lib.linux-x86_64-3.6/apex_C.cpython-36m-x86_64-linux-gnu.so
    building 'amp_C' extension
    gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/amp_C_frontend.cpp -o build/temp.linux-x86_64-3.6/csrc/amp_C_frontend.o -O3 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=amp_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
    /usr/local/cuda/bin/nvcc -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/multi_tensor_scale_kernel.cu -o build/temp.linux-x86_64-3.6/csrc/multi_tensor_scale_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -lineinfo -O3 --use_fast_math -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=amp_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    /usr/local/cuda/bin/nvcc -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/multi_tensor_axpby_kernel.cu -o build/temp.linux-x86_64-3.6/csrc/multi_tensor_axpby_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -lineinfo -O3 --use_fast_math -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=amp_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    /usr/local/cuda/bin/nvcc -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/multi_tensor_l2norm_kernel.cu -o build/temp.linux-x86_64-3.6/csrc/multi_tensor_l2norm_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -lineinfo -O3 --use_fast_math -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=amp_C -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    g++ -pthread -shared -B /opt/conda/compiler_compat -L/opt/conda/lib -Wl,-rpath=/opt/conda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/csrc/amp_C_frontend.o build/temp.linux-x86_64-3.6/csrc/multi_tensor_scale_kernel.o build/temp.linux-x86_64-3.6/csrc/multi_tensor_axpby_kernel.o build/temp.linux-x86_64-3.6/csrc/multi_tensor_l2norm_kernel.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.6/amp_C.cpython-36m-x86_64-linux-gnu.so
    building 'fused_adam_cuda' extension
    gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/fused_adam_cuda.cpp -o build/temp.linux-x86_64-3.6/csrc/fused_adam_cuda.o -O3 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fused_adam_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
    /usr/local/cuda/bin/nvcc -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/fused_adam_cuda_kernel.cu -o build/temp.linux-x86_64-3.6/csrc/fused_adam_cuda_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -O3 --use_fast_math -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fused_adam_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    g++ -pthread -shared -B /opt/conda/compiler_compat -L/opt/conda/lib -Wl,-rpath=/opt/conda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/csrc/fused_adam_cuda.o build/temp.linux-x86_64-3.6/csrc/fused_adam_cuda_kernel.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.6/fused_adam_cuda.cpython-36m-x86_64-linux-gnu.so
    building 'syncbn' extension
    gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/syncbn.cpp -o build/temp.linux-x86_64-3.6/csrc/syncbn.o -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=syncbn -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
    /usr/local/cuda/bin/nvcc -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/welford.cu -o build/temp.linux-x86_64-3.6/csrc/welford.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=syncbn -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    g++ -pthread -shared -B /opt/conda/compiler_compat -L/opt/conda/lib -Wl,-rpath=/opt/conda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/csrc/syncbn.o build/temp.linux-x86_64-3.6/csrc/welford.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.6/syncbn.cpython-36m-x86_64-linux-gnu.so
    building 'fused_layer_norm_cuda' extension
    gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/layer_norm_cuda.cpp -o build/temp.linux-x86_64-3.6/csrc/layer_norm_cuda.o -O3 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fused_layer_norm_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid for C/ObjC but not for C++
    /usr/local/cuda/bin/nvcc -I/opt/conda/lib/python3.6/site-packages/torch/lib/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/lib/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c csrc/layer_norm_cuda_kernel.cu -o build/temp.linux-x86_64-3.6/csrc/layer_norm_cuda_kernel.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --compiler-options '-fPIC' -maxrregcount=50 -O3 --use_fast_math -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fused_layer_norm_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
    g++ -pthread -shared -B /opt/conda/compiler_compat -L/opt/conda/lib -Wl,-rpath=/opt/conda/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/csrc/layer_norm_cuda.o build/temp.linux-x86_64-3.6/csrc/layer_norm_cuda_kernel.o -L/usr/local/cuda/lib64 -lcudart -o build/lib.linux-x86_64-3.6/fused_layer_norm_cuda.cpython-36m-x86_64-linux-gnu.so
    running install_lib
    creating /opt/conda/lib/python3.6/site-packages/apex
    creating /opt/conda/lib/python3.6/site-packages/apex/reparameterization
    copying build/lib.linux-x86_64-3.6/apex/reparameterization/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/reparameterization
    copying build/lib.linux-x86_64-3.6/apex/reparameterization/reparameterization.py -> /opt/conda/lib/python3.6/site-packages/apex/reparameterization
    copying build/lib.linux-x86_64-3.6/apex/reparameterization/weight_norm.py -> /opt/conda/lib/python3.6/site-packages/apex/reparameterization
    creating /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/sync_batchnorm_kernel.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/LARC.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/sync_batchnorm.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/distributed.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/optimized_sync_batchnorm.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/multiproc.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/parallel/optimized_sync_batchnorm_kernel.py -> /opt/conda/lib/python3.6/site-packages/apex/parallel
    copying build/lib.linux-x86_64-3.6/apex/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex
    creating /opt/conda/lib/python3.6/site-packages/apex/normalization
    copying build/lib.linux-x86_64-3.6/apex/normalization/fused_layer_norm.py -> /opt/conda/lib/python3.6/site-packages/apex/normalization
    copying build/lib.linux-x86_64-3.6/apex/normalization/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/normalization
    creating /opt/conda/lib/python3.6/site-packages/apex/fp16_utils
    copying build/lib.linux-x86_64-3.6/apex/fp16_utils/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/fp16_utils
    copying build/lib.linux-x86_64-3.6/apex/fp16_utils/fp16util.py -> /opt/conda/lib/python3.6/site-packages/apex/fp16_utils
    copying build/lib.linux-x86_64-3.6/apex/fp16_utils/loss_scaler.py -> /opt/conda/lib/python3.6/site-packages/apex/fp16_utils
    copying build/lib.linux-x86_64-3.6/apex/fp16_utils/fp16_optimizer.py -> /opt/conda/lib/python3.6/site-packages/apex/fp16_utils
    creating /opt/conda/lib/python3.6/site-packages/apex/optimizers
    copying build/lib.linux-x86_64-3.6/apex/optimizers/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/optimizers
    copying build/lib.linux-x86_64-3.6/apex/optimizers/fp16_optimizer.py -> /opt/conda/lib/python3.6/site-packages/apex/optimizers
    copying build/lib.linux-x86_64-3.6/apex/optimizers/fused_adam.py -> /opt/conda/lib/python3.6/site-packages/apex/optimizers
    creating /opt/conda/lib/python3.6/site-packages/apex/RNN
    copying build/lib.linux-x86_64-3.6/apex/RNN/models.py -> /opt/conda/lib/python3.6/site-packages/apex/RNN
    copying build/lib.linux-x86_64-3.6/apex/RNN/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/RNN
    copying build/lib.linux-x86_64-3.6/apex/RNN/RNNBackend.py -> /opt/conda/lib/python3.6/site-packages/apex/RNN
    copying build/lib.linux-x86_64-3.6/apex/RNN/cells.py -> /opt/conda/lib/python3.6/site-packages/apex/RNN
    creating /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/compat.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    creating /opt/conda/lib/python3.6/site-packages/apex/amp/lists
    copying build/lib.linux-x86_64-3.6/apex/amp/lists/torch_overrides.py -> /opt/conda/lib/python3.6/site-packages/apex/amp/lists
    copying build/lib.linux-x86_64-3.6/apex/amp/lists/tensor_overrides.py -> /opt/conda/lib/python3.6/site-packages/apex/amp/lists
    copying build/lib.linux-x86_64-3.6/apex/amp/lists/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/amp/lists
    copying build/lib.linux-x86_64-3.6/apex/amp/lists/functional_overrides.py -> /opt/conda/lib/python3.6/site-packages/apex/amp/lists
    copying build/lib.linux-x86_64-3.6/apex/amp/__version__.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/wrap.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/_initialize.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/handle.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/frontend.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/utils.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/opt.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/_process_optimizer.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/_amp_state.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/amp.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/rnn_compat.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    copying build/lib.linux-x86_64-3.6/apex/amp/scaler.py -> /opt/conda/lib/python3.6/site-packages/apex/amp
    creating /opt/conda/lib/python3.6/site-packages/apex/multi_tensor_apply
    copying build/lib.linux-x86_64-3.6/apex/multi_tensor_apply/multi_tensor_apply.py -> /opt/conda/lib/python3.6/site-packages/apex/multi_tensor_apply
    copying build/lib.linux-x86_64-3.6/apex/multi_tensor_apply/__init__.py -> /opt/conda/lib/python3.6/site-packages/apex/multi_tensor_apply
    copying build/lib.linux-x86_64-3.6/apex_C.cpython-36m-x86_64-linux-gnu.so -> /opt/conda/lib/python3.6/site-packages
    copying build/lib.linux-x86_64-3.6/syncbn.cpython-36m-x86_64-linux-gnu.so -> /opt/conda/lib/python3.6/site-packages
    copying build/lib.linux-x86_64-3.6/fused_adam_cuda.cpython-36m-x86_64-linux-gnu.so -> /opt/conda/lib/python3.6/site-packages
    copying build/lib.linux-x86_64-3.6/amp_C.cpython-36m-x86_64-linux-gnu.so -> /opt/conda/lib/python3.6/site-packages
    copying build/lib.linux-x86_64-3.6/fused_layer_norm_cuda.cpython-36m-x86_64-linux-gnu.so -> /opt/conda/lib/python3.6/site-packages
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/reparameterization/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/reparameterization/reparameterization.py to reparameterization.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/reparameterization/weight_norm.py to weight_norm.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/sync_batchnorm_kernel.py to sync_batchnorm_kernel.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/LARC.py to LARC.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/sync_batchnorm.py to sync_batchnorm.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/distributed.py to distributed.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/optimized_sync_batchnorm.py to optimized_sync_batchnorm.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/multiproc.py to multiproc.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/parallel/optimized_sync_batchnorm_kernel.py to optimized_sync_batchnorm_kernel.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/normalization/fused_layer_norm.py to fused_layer_norm.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/normalization/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/fp16_utils/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/fp16_utils/fp16util.py to fp16util.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/fp16_utils/loss_scaler.py to loss_scaler.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/fp16_utils/fp16_optimizer.py to fp16_optimizer.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/optimizers/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/optimizers/fp16_optimizer.py to fp16_optimizer.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/optimizers/fused_adam.py to fused_adam.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/RNN/models.py to models.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/RNN/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/RNN/RNNBackend.py to RNNBackend.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/RNN/cells.py to cells.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/compat.py to compat.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/lists/torch_overrides.py to torch_overrides.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/lists/tensor_overrides.py to tensor_overrides.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/lists/__init__.py to __init__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/lists/functional_overrides.py to functional_overrides.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/__version__.py to __version__.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/wrap.py to wrap.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/_initialize.py to _initialize.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/handle.py to handle.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/frontend.py to frontend.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/utils.py to utils.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/opt.py to opt.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/_process_optimizer.py to _process_optimizer.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/_amp_state.py to _amp_state.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/amp.py to amp.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/rnn_compat.py to rnn_compat.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/amp/scaler.py to scaler.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/multi_tensor_apply/multi_tensor_apply.py to multi_tensor_apply.cpython-36.pyc
    byte-compiling /opt/conda/lib/python3.6/site-packages/apex/multi_tensor_apply/__init__.py to __init__.cpython-36.pyc
    running install_egg_info
    running egg_info
    creating apex.egg-info
    writing apex.egg-info/PKG-INFO
    writing dependency_links to apex.egg-info/dependency_links.txt
    writing top-level names to apex.egg-info/top_level.txt
    writing manifest file 'apex.egg-info/SOURCES.txt'
    reading manifest file 'apex.egg-info/SOURCES.txt'
    writing manifest file 'apex.egg-info/SOURCES.txt'
    Copying apex.egg-info to /opt/conda/lib/python3.6/site-packages/apex-0.1-py3.6.egg-info
    running install_scripts
    writing list of installed files to '/tmp/pip-record-sz37cy8d/install-record.txt'
done
  Removing source in /tmp/pip-req-build-wajk9v9c
Successfully installed apex-0.1
Cleaning up...
Removed build tracker '/tmp/pip-req-tracker-5n8zf4po'
1 location(s) to search for versions of pip:
* https://pypi.org/simple/pip/
Getting page https://pypi.org/simple/pip/
Starting new HTTPS connection (1): pypi.org:443
Could not fetch URL https://pypi.org/simple/pip/: connection error: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/pip/ (Caused by NewConnectionError('<pip._vendor.urllib3.connection.VerifiedHTTPSConnection object at 0x7fb9fe6aeb38>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution',)) - skipping
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import operator 
import sys
from sklearn import metrics
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
%load_ext autoreload
%autoreload 2
%matplotlib inline
from tqdm import tqdm, tqdm_notebook
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil
device=torch.device('cuda')
MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 1
Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = "../input"
WORK_DIR = "../working/"
num_to_load=1000000                         #Train size to match time limit
valid_size= 100000                          #Validation Size
TOXICITY_COLUMN = 'target'
# Add the Bart Pytorch repo to the PATH
# using files from: https://github.com/huggingface/pytorch-pretrained-BERT
package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
/opt/conda/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 216, got 192
  return f(*args, **kwds)
/opt/conda/lib/python3.6/importlib/_bootstrap.py:219: ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__
  return f(*args, **kwds)
/opt/conda/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  return f(*args, **kwds)
# Translate model from tensorflow to pytorch
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
BERT_MODEL_PATH + 'bert_config.json',
WORK_DIR + 'pytorch_model.bin')

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')
Building PyTorch model from configuration: {
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}

Converting TensorFlow checkpoint from /kaggle/input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_model.ckpt
Loading TF weight bert/embeddings/LayerNorm/beta with shape [768]
Loading TF weight bert/embeddings/LayerNorm/gamma with shape [768]
Loading TF weight bert/embeddings/position_embeddings with shape [512, 768]
Loading TF weight bert/embeddings/token_type_embeddings with shape [2, 768]
Loading TF weight bert/embeddings/word_embeddings with shape [30522, 768]
Loading TF weight bert/encoder/layer_0/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_0/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_0/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_0/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_0/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_0/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_0/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_0/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_0/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_0/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_0/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_0/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_0/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_0/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_0/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_0/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_1/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_1/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_1/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_1/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_1/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_1/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_1/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_1/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_1/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_1/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_1/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_1/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_1/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_1/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_1/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_1/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_10/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_10/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_10/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_10/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_10/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_10/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_10/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_10/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_10/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_10/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_10/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_10/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_10/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_10/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_10/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_10/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_11/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_11/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_11/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_11/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_11/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_11/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_11/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_11/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_11/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_11/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_11/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_11/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_11/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_11/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_11/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_11/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_2/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_2/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_2/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_2/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_2/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_2/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_2/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_2/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_2/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_2/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_2/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_2/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_2/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_2/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_2/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_2/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_3/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_3/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_3/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_3/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_3/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_3/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_3/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_3/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_3/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_3/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_3/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_3/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_3/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_3/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_3/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_3/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_4/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_4/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_4/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_4/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_4/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_4/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_4/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_4/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_4/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_4/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_4/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_4/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_4/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_4/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_4/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_4/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_5/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_5/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_5/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_5/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_5/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_5/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_5/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_5/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_5/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_5/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_5/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_5/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_5/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_5/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_5/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_5/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_6/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_6/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_6/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_6/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_6/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_6/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_6/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_6/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_6/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_6/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_6/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_6/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_6/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_6/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_6/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_6/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_7/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_7/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_7/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_7/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_7/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_7/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_7/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_7/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_7/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_7/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_7/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_7/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_7/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_7/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_7/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_7/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_8/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_8/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_8/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_8/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_8/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_8/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_8/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_8/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_8/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_8/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_8/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_8/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_8/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_8/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_8/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_8/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/encoder/layer_9/attention/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_9/attention/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_9/attention/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_9/attention/output/dense/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_9/attention/self/key/bias with shape [768]
Loading TF weight bert/encoder/layer_9/attention/self/key/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_9/attention/self/query/bias with shape [768]
Loading TF weight bert/encoder/layer_9/attention/self/query/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_9/attention/self/value/bias with shape [768]
Loading TF weight bert/encoder/layer_9/attention/self/value/kernel with shape [768, 768]
Loading TF weight bert/encoder/layer_9/intermediate/dense/bias with shape [3072]
Loading TF weight bert/encoder/layer_9/intermediate/dense/kernel with shape [768, 3072]
Loading TF weight bert/encoder/layer_9/output/LayerNorm/beta with shape [768]
Loading TF weight bert/encoder/layer_9/output/LayerNorm/gamma with shape [768]
Loading TF weight bert/encoder/layer_9/output/dense/bias with shape [768]
Loading TF weight bert/encoder/layer_9/output/dense/kernel with shape [3072, 768]
Loading TF weight bert/pooler/dense/bias with shape [768]
Loading TF weight bert/pooler/dense/kernel with shape [768, 768]
Loading TF weight cls/predictions/output_bias with shape [30522]
Loading TF weight cls/predictions/transform/LayerNorm/beta with shape [768]
Loading TF weight cls/predictions/transform/LayerNorm/gamma with shape [768]
Loading TF weight cls/predictions/transform/dense/bias with shape [768]
Loading TF weight cls/predictions/transform/dense/kernel with shape [768, 768]
Loading TF weight cls/seq_relationship/output_bias with shape [2]
Loading TF weight cls/seq_relationship/output_weights with shape [2, 768]
Initialize PyTorch weight ['bert', 'embeddings', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'embeddings', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'embeddings', 'position_embeddings']
Initialize PyTorch weight ['bert', 'embeddings', 'token_type_embeddings']
Initialize PyTorch weight ['bert', 'embeddings', 'word_embeddings']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_0', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_1', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_10', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_11', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_2', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_3', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_4', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_5', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_6', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_7', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_8', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'self', 'key', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'self', 'key', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'self', 'query', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'self', 'query', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'self', 'value', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'attention', 'self', 'value', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'intermediate', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'intermediate', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'output', 'LayerNorm', 'beta']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'output', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'output', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'encoder', 'layer_9', 'output', 'dense', 'kernel']
Initialize PyTorch weight ['bert', 'pooler', 'dense', 'bias']
Initialize PyTorch weight ['bert', 'pooler', 'dense', 'kernel']
Initialize PyTorch weight ['cls', 'predictions', 'output_bias']
Initialize PyTorch weight ['cls', 'predictions', 'transform', 'LayerNorm', 'beta']
Initialize PyTorch weight ['cls', 'predictions', 'transform', 'LayerNorm', 'gamma']
Initialize PyTorch weight ['cls', 'predictions', 'transform', 'dense', 'bias']
Initialize PyTorch weight ['cls', 'predictions', 'transform', 'dense', 'kernel']
Initialize PyTorch weight ['cls', 'seq_relationship', 'output_bias']
Initialize PyTorch weight ['cls', 'seq_relationship', 'output_weights']
Save PyTorch model to ../working/pytorch_model.bin
'../working/bert_config.json'
os.listdir("../working")
['bert_config.json',
 '__notebook__.ipynb',
 'pytorch_model.bin',
 '__output__.json']
# This is the Bert configuration file
from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'+'bert_config.json')
# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
%%time
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
train_df = pd.read_csv(os.path.join(Data_dir,"train.csv")).sample(num_to_load+valid_size,random_state=SEED)
print('loaded %d records' % len(train_df))

# Make sure all comment_text values are strings
train_df['comment_text'] = train_df['comment_text'].astype(str) 

sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)
train_df=train_df.fillna(0)
# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']

train_df = train_df.drop(['comment_text'],axis=1)
# convert target to 0,1
train_df['target']=(train_df['target']>=0.5).astype(float)
loaded 1100000 records
X = sequences[:num_to_load]                
y = train_df[y_columns].values[:num_to_load]
X_val = sequences[num_to_load:]                
y_val = train_df[y_columns].values[num_to_load:]
test_df=train_df.tail(valid_size).copy()
train_df=train_df.head(num_to_load)
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))
output_model_file = "bert_pytorch.bin"

lr=2e-5
batch_size = 32
accumulation_steps=1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained("../working",cache_dir=None,num_labels=len(y_columns))
model.zero_grad()
model = model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
train = train_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
model=model.train()

tq = tqdm_notebook(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)
    for i,(x_batch, y_batch) in tk0:
        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)


torch.save(model.state_dict(), output_model_file)
<torch._C.Generator at 0x7f516467cdb0>
# Run validation
# The following 2 lines are not needed but show how to download the model for prediction
model = BertForSequenceClassification(bert_config,num_labels=len(y_columns))
model.load_state_dict(torch.load(output_model_file ))
model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (2): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (3): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (4): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (5): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (6): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (7): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (8): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (9): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (10): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=1, bias=True)
)
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (2): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (3): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (4): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (5): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (6): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (7): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (8): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (9): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (10): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): FusedLayerNorm(torch.Size([768]), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=1, bias=True)
)
# From baseline kernel

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]>0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)



SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
MODEL_NAME = 'model1'
test_df[MODEL_NAME]=torch.sigmoid(torch.tensor(valid_preds)).numpy()
TOXICITY_COLUMN = 'target'
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, 'target')
bias_metrics_df
get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))
	bnsp_auc	bpsn_auc	subgroup	subgroup_auc	subgroup_size
2	0.970597	0.848254	homosexual_gay_or_lesbian	0.839271	579
7	0.972843	0.851686	white	0.847527	1312
6	0.978213	0.829808	black	0.856715	744
5	0.966057	0.894989	muslim	0.874820	1076
4	0.958657	0.922020	jewish	0.896079	387
1	0.960453	0.944828	female	0.925648	2790
3	0.951218	0.955274	christian	0.925815	1997
0	0.968533	0.932147	male	0.928682	2260
8	0.980000	0.930358	psychiatric_or_mental_illness	0.954240	227
