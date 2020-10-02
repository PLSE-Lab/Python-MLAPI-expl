#!/usr/bin/env python
# coding: utf-8

# # Overview
# 

# ### Original Task
# The provided file contains a simple kernel and full code to measure its
# performance, please optimize this kernel to maximize its performance.
# 
# The data dimensions are to remain the same, i.e. you can assume that they
# will not change during our testing and that it is acceptable for your
# optimizations to rely upon them. You can change the kernel itself in any way
# you like, naturally the mathematical operation for each output element must
# be the same.
# 
# You can also change the kernel launch configuration in any way you like,
# i.e. you can select the grid and block dimensions as you see fit. However,
# do not modify the for-loop that launches the kernel 'nreps' times (line 40)
# since this loop is simply there to average the elapsed time over several
# launches.
# 
# Please provide a brief summary of the optimizations you apply, a sentence or
# two for each optimization is sufficient.
# 
# 
# 

# # Original Implementation
# Here is the original implementation as supplied in the file

# In[ ]:


get_ipython().run_cell_magic('file', 'cuda_prog.cu', '#include <stdio.h>\n\n__global__ void kernel_A( float *g_data, int dimx, int dimy, int niterations )\n{\n\tint ix  = blockIdx.x;\n\tint iy  = blockIdx.y*blockDim.y + threadIdx.y;\n\tint idx = iy*dimx + ix;\n\n\tfloat value = g_data[idx];\n\n\tif( ix % 2 )\n\t{\n\t\tfor(int i=0; i<niterations; i++)\n\t\t{\n\t\t\tvalue += sqrtf( logf(value) + 1.f );\n\t\t}\n\t}\n\telse\n\t{\n\t\tfor(int i=0; i<niterations; i++)\n\t\t{\n\t\t\tvalue += sqrtf( cosf(value) + 1.f );\n\t\t}\n\t}\n\n\tg_data[idx] = value;\n}\n\nfloat timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )\n{\n\tfloat elapsed_time_ms=0.0f;\n\tcudaEvent_t start, stop;\n\tcudaEventCreate( &start );\n\tcudaEventCreate( &stop  );\n\n\tdim3 block( blockx, blocky );\n\tdim3 grid( dimx/block.x, dimy/block.y );\n\n\tcudaEventRecord( start, 0 );\n\tfor(int i=0; i<nreps; i++)\t// do not change this loop, it\'s not part of the algorithm - it\'s just to average time over several kernel launches\n\t\tkernel<<<grid,block>>>( d_data, dimx,dimy, niterations );\n\tcudaEventRecord( stop, 0 );\n\tcudaDeviceSynchronize();\n\tcudaEventElapsedTime( &elapsed_time_ms, start, stop );\n\telapsed_time_ms /= nreps;\n\n\tcudaEventDestroy( start );\n\tcudaEventDestroy( stop );\n\n\treturn elapsed_time_ms;\n}\n\nint main()\n{\n\tint dimx = 2*1024;\n\tint dimy = 2*1024;\n\n\tint nreps = 10;\n\tint niterations = 20;\n\n\tint nbytes = dimx*dimy*sizeof(float);\n\n\tfloat *d_data=0, *h_data=0;\n\tcudaMalloc( (void**)&d_data, nbytes );\n\tif( 0 == d_data )\n\t{\n\t\tprintf("couldn\'t allocate GPU memory\\n");\n\t\treturn -1;\n\t}\n\tprintf("allocated %.2f MB on GPU\\n", nbytes/(1024.f*1024.f) );\n\th_data = (float*)malloc( nbytes );\n\tif( 0 == h_data )\n\t{\n\t\tprintf("couldn\'t allocate CPU memory\\n");\n\t\treturn -2;\n\t}\n\tprintf("allocated %.2f MB on CPU\\n", nbytes/(1024.f*1024.f) );\n\tfor(int i=0; i<dimx*dimy; i++)\n\t\th_data[i] = 10.f + rand() % 256;\n\tcudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );\n\n\tfloat elapsed_time_ms=0.0f;\n\n\telapsed_time_ms = timing_experiment( kernel_A, d_data, dimx,dimy, niterations, nreps, 1, 256 );\n\tprintf("A:  %8.2f ms\\n", elapsed_time_ms );\n\n\tprintf("CUDA: %s\\n", cudaGetErrorString( cudaGetLastError() ) );\n\n\tif( d_data )\n\t\tcudaFree( d_data );\n\tif( h_data )\n\t\tfree( h_data );\n\n\tcudaDeviceReset();\n\n\treturn 0;\n}')


# In[ ]:


get_ipython().system('nvcc -o original_version cuda_prog.cu -Wno-deprecated-gpu-targets')


# In[ ]:


get_ipython().system('./original_version')


# # Varying Block Sizes
# 
# In order to get a better idea how the model performs we have to move beyond the fixed block sizes of `1x256` and adding some code to make sure our output still matches within $<10^{-5}$ tolerance. I use temporary files and templating to generate and run a number of different parameters and display the results

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 72
plt.rcParams["font.size"] = 11
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better


# In[ ]:


import pandas as pd
import numpy as np
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from subprocess import check_output, STDOUT
import re

from itertools import product
from tqdm import tqdm
tqdm_product = lambda *args: tqdm(list(product(*args)))

def build_and_run(in_code, exec_prefix='', verbose=False):
  with TemporaryDirectory() as tmp_dir:
    out_file = os.path.join(tmp_dir, 'run_model')
    with NamedTemporaryFile(suffix='.cu', mode='w', delete=not verbose) as c_file:
      c_file.write(in_code)
      c_file.flush()
      check_output(f'nvcc -o {out_file} {c_file.name}', 
                  stderr=STDOUT,
                  shell=True)
    out_msg = check_output(f'{exec_prefix} {out_file}'.strip(), shell=True, stderr=STDOUT).decode()
    if verbose:
      print(out_msg)
  out_lines = [re.split('\s+', x.strip()) for x in out_msg.split('\n') if len(x.strip())>0]
  return out_lines


# In[ ]:


CUDA_TEMPLATE = r"""
#include <stdio.h>

__global__ void kernel_A_ref( float *g_data, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + ix;

	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

__global__ void kernel_A_opt( float *g_data, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x*blockDim.x + threadIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + ix;

	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

float timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, dimx,dimy, niterations );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

int main()
{
	int dimx = 2*1024;
	int dimy = 2*1024;

	int nreps = 10;
	int niterations = 20;

	int nbytes = dimx*dimy*sizeof(float);
  
	float *d_data=0, *h_data=0, *opt_output=0, *ref_output=0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
  
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  opt_output = (float*)malloc( nbytes );
	if( 0 == opt_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  ref_output = (float*)malloc( nbytes );
	if( 0 == ref_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", 3*nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 10.f + rand() % 256;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  
  
  printf("blocks %02d %02d\n", {block_x}, {block_y});
  
	float elapsed_time_ms;
  
	elapsed_time_ms = timing_experiment( kernel_A_ref, d_data, dimx,dimy, niterations, nreps, 1, 256 );
	printf("A_ref:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy( ref_output, d_data, nbytes, cudaMemcpyDeviceToHost );
  
  
  // reset for optimized version
  if( d_data )
		cudaFree( d_data );
  cudaDeviceReset();
  
  cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  elapsed_time_ms = timing_experiment( kernel_A_opt, d_data, dimx,dimy, niterations, nreps, {block_x}, {block_y} );
	printf("A_opt:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy( opt_output, d_data, nbytes, cudaMemcpyDeviceToHost );
	// Verify precision of result
  int nb_correct_precisions = 0;
  const double precision    = 1e-6; // precision error max
  for (int i=0; i<dimx*dimy; ++i) {
      if (abs(ref_output[i] - opt_output[i]) <= precision) {
          nb_correct_precisions++;
      }
  }
  double score = nb_correct_precisions*100.0/(dimx*dimy);
  
	printf("CUDA: %s, Accuracy: %2.1f\n", cudaGetErrorString( cudaGetLastError() ) , score);

	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );

	cudaDeviceReset();

	return 0;
}
"""


# In[ ]:


def fancy_format(in_str, **kwargs):
    new_str = in_str.replace('{', '{{').replace('}', '}}')
    for key in kwargs.keys():
        new_str = new_str.replace('{{%s}}' % key, '{%s}' % key)
    return new_str.format(**kwargs)
cuda_code = lambda block_x=1, block_y=256: fancy_format(CUDA_TEMPLATE, block_x=block_x, block_y=block_y)


# ### Test Code

# In[ ]:


build_and_run(cuda_code(2, 128), exec_prefix='nvprof --print-gpu-trace', verbose=True);


# In[ ]:


build_and_run(cuda_code(1, 1))


# In[ ]:


val_range = [2**x for x in range(0, 11)]
# val_range = [1, 256] # minirun
test_reps = 5 # results are pretty unstable


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cuda_bench = [build_and_run(cuda_code(x, y)) for x,y, _ in \n              tqdm_product(val_range, val_range, range(test_reps))]')


# In[ ]:


bench_df = pd.DataFrame([{'block_x': int(x[2][1]),
               'block_y': int(x[2][2]),
               'time_optimized_ms': float(x[-2][1]),
               'time_ref_ms': float(x[-3][1]),
                         'accuracy': float(x[-1][-1])}
  for x in cuda_bench])
bench_df['fps_opt'] = 1000/bench_df['time_optimized_ms']
bench_df['fps_ref'] = 1000/bench_df['time_ref_ms']
bench_df.head(10)


# In[ ]:


bench_df['time_ref_ms'].hist(bins=30, figsize=(3,3))


# ## Show Accuracy
# We show the accuracy for each configuration since the results need to match

# In[ ]:


bench_df.pivot_table(index='block_x', columns='block_y', values='accuracy', aggfunc='median')


# ## Show FPS
# FPS is a nice number to visualize

# In[ ]:


bench_grid = bench_df.query('time_optimized_ms>0.0').pivot_table(index='block_x', 
                                                           columns='block_y', 
                                                           values='fps_opt',
                                                          aggfunc='median')
sns.heatmap(bench_grid, fmt='2.0f', annot=True, cmap='viridis')
bench_grid


# In[ ]:


bench_df.query('time_optimized_ms>0.0').  pivot_table(index='block_x', columns='block_y', values='time_ref_ms', aggfunc='median')


# # Two Kernels
# To avoid branching within the kernel we can use two kernels

# In[ ]:


CUDA_TWO_TEMPLATE = r"""
#include <stdio.h>

__global__ void kernel_A_ref( float *g_data, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + ix;

	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

__global__ void kernel_A_opt_log( float *g_data, int dimx, int dimy, int niterations)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + (2*ix+1);

	float value = g_data[idx];
  
  for(int i=0; i<niterations; i++)
  {
    value += sqrtf( logf(value) + 1.f );
  }
	g_data[idx] = value;
}

__global__ void kernel_A_opt_cos( float *g_data, int dimx, int dimy, int niterations)
{
	int ix  = blockIdx.x*blockDim.x + threadIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + (2*ix);
	float value = g_data[idx];

  for(int i=0; i<niterations; i++)
  {
    value += sqrtf( cosf(value) + 1.f );
  }
	g_data[idx] = value;
}

float timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, dimx,dimy, niterations );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

float timing_experiment_opt(float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/2/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		{
    kernel_A_opt_log<<<grid,block>>>( d_data, dimx,dimy, niterations );
    kernel_A_opt_cos<<<grid,block>>>( d_data, dimx,dimy, niterations );
    }
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

int main()
{
	int dimx = 2*1024;
	int dimy = 2*1024;

	int nreps = 10;
	int niterations = 20;

	int nbytes = dimx*dimy*sizeof(float);
  
	float *d_data=0, *h_data=0, *opt_output=0, *ref_output=0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
  
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  opt_output = (float*)malloc( nbytes );
	if( 0 == opt_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  ref_output = (float*)malloc( nbytes );
	if( 0 == ref_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", 3*nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 10.f + rand() % 256;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  
  
  printf("blocks %02d %02d\n", {block_x}, {block_y});
  
	float elapsed_time_ms;
  
	elapsed_time_ms = timing_experiment( kernel_A_ref, d_data, dimx,dimy, niterations, nreps, 1, 256 );
	printf("A_ref:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy( ref_output, d_data, nbytes, cudaMemcpyDeviceToHost );
  
  
  // reset for optimized version
  if( d_data )
		cudaFree( d_data );
  cudaDeviceReset();
  
  cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  elapsed_time_ms = timing_experiment_opt(d_data, dimx,dimy, niterations, nreps, {block_x}, {block_y} );
	printf("A_opt:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy( opt_output, d_data, nbytes, cudaMemcpyDeviceToHost );
	// Verify precision of result
  int nb_correct_precisions = 0;
  const double precision    = 1e-6; // precision error max
  for (int i=0; i<dimx*dimy; ++i) {
      if (abs(ref_output[i] - opt_output[i]) <= precision) {
          nb_correct_precisions++;
      }
  }
  double score = nb_correct_precisions*100.0/(dimx*dimy);
  
	printf("CUDA: %s, Accuracy: %2.1f\n", cudaGetErrorString( cudaGetLastError() ) , score);

	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );

	cudaDeviceReset();

	return 0;
}
"""


# In[ ]:


cuda_two_code = lambda block_x=1, block_y=256: fancy_format(CUDA_TWO_TEMPLATE, 
                                                            block_x=block_x, block_y=block_y)


# ### Test Two Kernel Output

# In[ ]:


build_and_run(cuda_two_code(8, 128), exec_prefix='nvprof --print-gpu-trace', verbose=True);


# In[ ]:


cuda_two_bench = [build_and_run(cuda_two_code(x, y)) for x,y, _ in 
                  tqdm_product(val_range, val_range, range(test_reps))]


# In[ ]:


bench_two_df = pd.DataFrame([{'block_x': int(x[2][1]),
               'block_y': int(x[2][2]),
               'time_optimized_ms': float(x[-2][1]),
               'time_ref_ms': float(x[-3][1]),
                         'accuracy': float(x[-1][-1])}
  for x in cuda_two_bench])
bench_two_df['fps_opt'] = 1000/bench_two_df['time_optimized_ms']
bench_two_df['fps_ref'] = 1000/bench_two_df['time_ref_ms']
bench_two_df.head(10)


# ### Accuracy

# In[ ]:


bench_grid = bench_two_df.query('time_optimized_ms>0.0').pivot_table(index='block_x', 
                                                           columns='block_y', 
                                                           values='fps_opt',
                                                          aggfunc='median')
sns.heatmap(bench_grid, fmt='2.0f', annot=True, cmap='viridis')
bench_grid


# In[ ]:


bench_grid = bench_two_df.query('time_optimized_ms>0.0').  pivot_table(index='block_x', columns='block_y', values='fps_opt', aggfunc='median')
sns.heatmap(bench_grid, fmt='2.0f', annot=True, cmap='viridis')
bench_grid


# # Lookup Table
# Since the original image only has 255 unique values, we can save quite a bit of time by just calculating the output for each unique input in a table and then mapping it. 
# 
# The initial values are generated by 
# ```
# h_data[i] = 10.f + rand() % 256;
# ```
# This means there are unique values from $[10, 265]$ and each one of them can be precomputed (including the niterations) for the `sqrt(log(` and `sqrt(cos(` functions. 
# ### Note
# Since the function is run 10 ($nrep$) times the values deviate from their initial values (since the output modifies the input) and thus the accuracy measurement is only meaningful for $nrep=1$

# In[ ]:


CUDA_LUT_TEMPLATE = r"""
#include <stdio.h>

__global__ void kernel_A_ref( float *g_data, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + ix;

	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

float timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, dimx,dimy, niterations );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}
__global__ void kernel_compute_lut(float *lut, int max_lut_val, int niterations) {
  int ix = threadIdx.x;
  
  if(blockIdx.x==0)
	{
    float value = ix+10;
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
    lut[ix+max_lut_val] = value;
	}
	else
	{
    float value = ix+10;
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
    lut[ix] = value;
	}

} 
__global__ void kernel_apply_lut( float *g_data, int dimx, int dimy, float *lut, int max_lut_val)
{
	int ix  = blockIdx.x*blockDim.x + threadIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	if ((ix<dimx) && (iy<dimy)) {
		int idx = iy*dimx + ix;
		int lut_idx = round(g_data[idx]-10);
		if (lut_idx<0) 
			lut_idx=0;
		if (lut_idx>=max_lut_val) 
			lut_idx=max_lut_val-1;
		
		if( ix % 2 )
			lut_idx+=max_lut_val;
		g_data[idx] = lut[lut_idx];
	}
}

float timing_experiment_opt(float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );
  float *lut = 0;
  const int max_lut_val = 255;
  cudaMalloc( (void**)&lut, 2*max_lut_val*sizeof(float));
	if( 0 == lut )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}

	dim3 img_block( blockx, blocky );
	dim3 img_grid( dimx/img_block.x, dimy/img_block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
	{
		kernel_compute_lut<<<2,max_lut_val>>>(lut, max_lut_val, niterations);
        kernel_apply_lut<<<img_grid,img_block>>>( d_data, dimx, dimy, lut, max_lut_val);
	}
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}


int main()
{
	int dimx = 2*1024;
	int dimy = 2*1024;

	int nreps = {nreps};
	int niterations = 20;

	int nbytes = dimx*dimy*sizeof(float);
  
	float *d_data=0, *h_data=0, *opt_output=0, *ref_output=0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
  
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  opt_output = (float*)malloc( nbytes );
	if( 0 == opt_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  ref_output = (float*)malloc( nbytes );
	if( 0 == ref_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", 3*nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 10.f + rand() % 256;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  
  printf("blocks %02d %02d\n", {block_x}, {block_y});
  
	float elapsed_time_ms;
  if ({run_ref}>0) {
		elapsed_time_ms = timing_experiment( kernel_A_ref, d_data, dimx,dimy, niterations, nreps, 1, 256 );
	} else {
		elapsed_time_ms = 1.6;
	}
	
	printf("A_ref:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy( ref_output, d_data, nbytes, cudaMemcpyDeviceToHost );
  
  
  // reset for optimized version
  if( d_data )
		cudaFree( d_data );
  cudaDeviceReset();
  
  cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  elapsed_time_ms = timing_experiment_opt(d_data, dimx, dimy, niterations, nreps, {block_x}, {block_y});
	printf("A_opt:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy( opt_output, d_data, nbytes, cudaMemcpyDeviceToHost );
	// Verify precision of result
  int nb_correct_precisions = 0;
  double mean_difference = 0;
  const double precision    = 1e-5; // precision error max
  for (int i=0; i<dimx*dimy; ++i) {
      if (abs(ref_output[i] - opt_output[i]) <= precision) {
          nb_correct_precisions++;
      }
      mean_difference += ref_output[i] - opt_output[i];
  }
  mean_difference/=dimx*dimy;
  double score = nb_correct_precisions*100.0/(dimx*dimy);
  
	printf("CUDA: %s, Error: %2.1f, Accuracy: %2.1f\n", 
    cudaGetErrorString( cudaGetLastError() ) , mean_difference, score);

	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );

	cudaDeviceReset();

	return 0;
}
"""


# In[ ]:


cuda_lut_code = lambda block_x=1, block_y=256, nreps=10, run_ref=1: fancy_format(CUDA_LUT_TEMPLATE, block_x=block_x, block_y=block_y, nreps=nreps, run_ref=run_ref)


# ### Test Code

# In[ ]:


build_and_run(cuda_lut_code(1, 128, nreps=1, run_ref=1), exec_prefix='nvprof --print-gpu-trace', verbose=True);


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cuda_lut_bench = [build_and_run(cuda_lut_code(x, y)) for x,y, _ in \n              tqdm_product(val_range, val_range, range(test_reps))]')


# In[ ]:


bench_lut_df = pd.DataFrame([{'block_x': int(x[2][1]),
               'block_y': int(x[2][2]),
               'time_optimized_ms': float(x[-2][1]),
               'time_ref_ms': float(x[-3][1]),
               'accuracy': float(x[-1][-1]),
               'error_msg': ' '.join(x[-1][1:3])}
  for x in cuda_lut_bench])
bench_lut_df['fps_opt'] = 1000/bench_lut_df['time_optimized_ms']
bench_lut_df['fps_ref'] = 1000/bench_lut_df['time_ref_ms']
bench_lut_df.head(10)


# ### FPS

# In[ ]:


bench_grid = bench_lut_df.query('error_msg=="no error,"').pivot_table(index='block_x', 
                                                           columns='block_y', 
                                                           values='fps_opt',
                                                          aggfunc='median')
sns.heatmap(bench_grid, fmt='2.0f', annot=True, cmap='viridis')
bench_grid


# ## Export the best version

# In[ ]:


bench_lut_df.  query('error_msg=="no error,"').  groupby(['block_x', 'block_y']).  agg({'time_optimized_ms': 'median'}).  reset_index().  sort_values('time_optimized_ms').  head(2)


# In[ ]:


try:
  from google.colab.files import download as FileLink
except: 
  from IPython.display import FileLink
with open('best_version.cu', 'w') as f:
  f.write(cuda_lut_code(128, 1, nreps=10, run_ref=1))
FileLink('best_version.cu')


# In[ ]:


build_and_run(cuda_lut_code(128, 1, nreps=10, run_ref=1), exec_prefix='nvprof --print-gpu-trace', verbose=True);


# # Pitched Arrays
# 
# Since the arrays are 2D we can try using pitched arrays but since the operation is pixel independent it probably doesn't make a difference

# In[ ]:


CUDA_PITCHED_TEMPLATE = r"""
#include <stdio.h>
__global__ void kernel_A_ref( float *g_data, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + ix;

	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

__global__ void kernel_A_opt( float *g_data, int d_pitch, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x*blockDim.x + threadIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*d_pitch + ix;
  
	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}
    
	g_data[idx] = value;
}



float timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );
  
  if (dimx % block.x != 0) grid.x += 1;
  if (dimy % block.y != 0) grid.y += 1;

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, dimx,dimy, niterations );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

float timing_experiment_pitch( void (*kernel)( float*, int, int,int,int), float *d_data, int d_pitch, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );
  
  if (dimx % block.x != 0) grid.x += 1;
  if (dimy % block.y != 0) grid.y += 1;

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, d_pitch, dimx, dimy, niterations );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

int main()
{
	const int dimx = 2*1024;
	const int dimy = 2*1024;

	const int nreps = 10;
	const int niterations = 20;

	int nbytes = dimx*dimy*sizeof(float);

	float *d_data=0, *h_data=0, *opt_output=0, *ref_output=0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
  
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  opt_output = (float*)malloc( nbytes );
	if( 0 == opt_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  ref_output = (float*)malloc( nbytes );
	if( 0 == ref_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", 3*nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 10.f + rand() % 256;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  
  printf("blocks %02d %02d\n", {block_x}, {block_y});
	float elapsed_time_ms=0.0f;

	elapsed_time_ms = timing_experiment( kernel_A_ref, d_data, dimx, dimy, niterations, nreps, 1, 256);
	printf("A_ref:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy(ref_output, d_data, nbytes, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
  
  // reseting CUDA memory
  if( d_data )
		cudaFree( d_data );
  
  cudaDeviceReset();
  
  
  // optimized approach
  const unsigned int size_of_float = sizeof(float);
  size_t  d_pitch_in_bytes;
  cudaMallocPitch((void**)&d_data, &d_pitch_in_bytes, dimx*size_of_float, dimy);
  
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  size_t d_pitch = d_pitch_in_bytes / size_of_float;
  cudaMemcpy2D(d_data, d_pitch_in_bytes,  h_data,   dimx * size_of_float,   dimx * size_of_float, dimy, cudaMemcpyHostToDevice);
  elapsed_time_ms = timing_experiment_pitch( kernel_A_opt, d_data, d_pitch, dimx, dimy, niterations, nreps, {block_x}, {block_y});
  printf("A_opt:  %8.2f ms\n", elapsed_time_ms );
  cudaMemcpy2D(opt_output,  dimx * size_of_float, d_data,  d_pitch_in_bytes,  dimx * size_of_float, dimy, cudaMemcpyDeviceToHost);
  // Verify precision of result
  int nb_correct_precisions = 0;
  const double precision    = 1e-6; // precision error max
  for (int i=0; i<dimx*dimy; ++i) {
      if (abs(ref_output[i] - opt_output[i]) <= precision) {
          nb_correct_precisions++;
      }
  }
  double score = nb_correct_precisions*100.0/(dimx*dimy);
  
	printf("CUDA: %s, Accuracy: %2.1f\n", cudaGetErrorString( cudaGetLastError() ) , score);
  
	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );
    
  if( opt_output )
		free( opt_output );
  
  if( ref_output )
		free( ref_output );

	cudaDeviceReset();

	return 0;
}
"""


# In[ ]:


cuda_pitched_code = lambda block_x=1, block_y=256: fancy_format(CUDA_PITCHED_TEMPLATE, block_x=block_x, block_y=block_y)


# In[ ]:


build_and_run(cuda_pitched_code(16, 16), exec_prefix='nvprof --print-gpu-trace', verbose=True);


# In[ ]:


cuda_pitched_bench = [build_and_run(cuda_pitched_code(x, y)) for x,y, _ in 
                      tqdm_product(val_range, val_range, range(test_reps))]


# ### Accuracy

# In[ ]:


bench_pitched_df = pd.DataFrame([{'block_x': int(x[2][1]),
               'block_y': int(x[2][2]),
               'time_optimized_ms': float(x[-2][1]),
               'time_ref_ms': float(x[-3][1]),
                         'accuracy': float(x[-1][-1])}
  for x in cuda_pitched_bench])
bench_pitched_df['fps_opt'] = 1000/bench_pitched_df['time_optimized_ms']
bench_pitched_df['fps_ref'] = 1000/bench_pitched_df['time_ref_ms']
bench_pitched_df.pivot_table(index='block_x', columns='block_y', values='accuracy', aggfunc='median')


# ### FPS

# In[ ]:


bench_grid = bench_pitched_df.query('time_optimized_ms>0.0').  pivot_table(index='block_x', columns='block_y', values='fps_opt', aggfunc='median')
sns.heatmap(bench_grid, fmt='2.0f', annot=True, cmap='viridis')
bench_grid


# # Shared Memory
# 
# We can try having thread shared memory to group all the reading and writing and once, but this probably also won't help much

# In[ ]:


CUDA_SHARED_CODE = r"""
#include <stdio.h>
__global__ void kernel_A_ref( float *g_data, int dimx, int dimy, int niterations )
{
	int ix  = blockIdx.x;
	int iy  = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = iy*dimx + ix;

	float value = g_data[idx];

	if( ix % 2 )
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}

	g_data[idx] = value;
}

__global__ void kernel_A_opt( float *g_data, int d_pitch, int dimx, int dimy, int niterations )
{
  __shared__ float shared_A[{block_x}][{block_y}];

  __shared__ int start_x;
  __shared__ int start_y;
  
  start_x = blockDim.x * blockIdx.x;
  start_y = blockDim.y * blockIdx.y;
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
	int ix  = start_x + tx;
	int iy  = start_y + ty;
  
  const int mode = ix % 2;
  
  // load data
  if ((iy + ty < dimy) && (ix+tx<dimx)) {
      shared_A[tx][ty] = g_data[iy*d_pitch + ix];
  } else {
      shared_A[tx][ty] = 0;
  }
  
	//__syncthreads();
  float value = shared_A[tx][ty];
	if(mode)
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( logf(value) + 1.f );
		}
	}
	else
	{
		for(int i=0; i<niterations; i++)
		{
			value += sqrtf( cosf(value) + 1.f );
		}
	}
  shared_A[tx][ty] = value;
  //__syncthreads();
  if ((iy + ty < dimy) && (ix+tx<dimx))
      g_data[iy*d_pitch + ix] = shared_A[tx][ty];
}



float timing_experiment( void (*kernel)( float*, int,int,int), float *d_data, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );
  
  if (dimx % block.x != 0) grid.x += 1;
  if (dimy % block.y != 0) grid.y += 1;

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, dimx,dimy, niterations );
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

float timing_experiment_pitch( void (*kernel)( float*, int, int,int,int), float *d_data, int d_pitch, int dimx, int dimy, int niterations, int nreps, int blockx, int blocky )
{
	float elapsed_time_ms=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++) {	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel<<<grid,block>>>( d_data, d_pitch, dimx, dimy, niterations );
  }
	cudaEventRecord( stop, 0 );
	cudaDeviceSynchronize();
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );
	elapsed_time_ms /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return elapsed_time_ms;
}

int main()
{
	const int dimx = 2*1024;
	const int dimy = 2*1024;

	const int nreps = 10;
	const int niterations = 20;

	int nbytes = dimx*dimy*sizeof(float);

	float *d_data=0, *h_data=0, *opt_output=0, *ref_output=0;
	cudaMalloc( (void**)&d_data, nbytes );
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
  
	h_data = (float*)malloc( nbytes );
	if( 0 == h_data )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  opt_output = (float*)malloc( nbytes );
	if( 0 == opt_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
  ref_output = (float*)malloc( nbytes );
	if( 0 == ref_output )
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", 3*nbytes/(1024.f*1024.f) );
	for(int i=0; i<dimx*dimy; i++)
		h_data[i] = 10.f + rand() % 256;
	cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );
  
  printf("blocks %02d %02d\n", {block_x}, {block_y});
	float elapsed_time_ms=0.0f;

	elapsed_time_ms = timing_experiment( kernel_A_ref, d_data, dimx, dimy, niterations, nreps, 1, 256);
	printf("A_ref:  %8.2f ms\n", elapsed_time_ms );
  
  cudaMemcpy(ref_output, d_data, nbytes, cudaMemcpyDeviceToHost );
  cudaDeviceSynchronize();
  
  // reseting CUDA memory
  if( d_data )
		cudaFree( d_data );
  
  cudaDeviceReset();
  
  // optimized approach
  const unsigned int size_of_float = sizeof(float);
  size_t  d_pitch_in_bytes;
  cudaMallocPitch((void**)&d_data, &d_pitch_in_bytes, dimx*size_of_float, dimy);
  
	if( 0 == d_data )
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
  size_t d_pitch = d_pitch_in_bytes / size_of_float;
  cudaMemcpy2D(d_data, d_pitch_in_bytes,  h_data,   dimx * size_of_float,   dimx * size_of_float,   dimy, cudaMemcpyHostToDevice);
  elapsed_time_ms = timing_experiment_pitch( kernel_A_opt, d_data, d_pitch, dimx, dimy, niterations, nreps, {block_x}, {block_y});
  printf("A_opt:  %8.2f ms\n", elapsed_time_ms );
  cudaMemcpy2D(opt_output,  dimx * size_of_float, d_data,  d_pitch_in_bytes,  dimx * size_of_float, dimy, cudaMemcpyDeviceToHost);
  // Verify precision of result
  int nb_correct_precisions = 0;
  const double precision    = 1e-6; // precision error max
  for (int i=0; i<dimx*dimy; ++i) {
      if (abs(ref_output[i] - opt_output[i]) <= precision) {
          nb_correct_precisions++;
      }
  }
  double score = nb_correct_precisions*100.0/(dimx*dimy);
  
	printf("CUDA: %s, Accuracy: %2.1f\n", cudaGetErrorString( cudaGetLastError() ) , score);
  
	if( d_data )
		cudaFree( d_data );
	if( h_data )
		free( h_data );
    
  if( opt_output )
		free( opt_output );
  
  if( ref_output )
		free( ref_output );

	cudaDeviceReset();

	return 0;
}
"""


# In[ ]:


cuda_shared_code = lambda block_x=2, block_y=256, block_dim=16: fancy_format(CUDA_SHARED_CODE, block_x=block_x, block_y=block_y, block_dim=block_dim)


# In[ ]:


build_and_run(cuda_shared_code(8, 128), exec_prefix='nvprof --print-gpu-trace', verbose=True);


# In[ ]:


cuda_shared_bench = [build_and_run(cuda_shared_code(x, y)) for x,y, _ in 
                     tqdm_product([1, 2], val_range, range(test_reps)) if x<=512]


# In[ ]:


bench_shared_df = pd.DataFrame([{'block_x': int(x[2][1]),
               'block_y': int(x[2][2]),
               'time_optimized_ms': float(x[-2][1]),
               'time_ref_ms': float(x[-3][1]),
                         'accuracy': float(x[-1][-1])}
  for x in cuda_shared_bench])
bench_shared_df['fps_opt'] = 1000/bench_shared_df['time_optimized_ms']
bench_shared_df['fps_ref'] = 1000/bench_shared_df['time_ref_ms']
bench_shared_df.head(10)


# In[ ]:


bench_shared_df.pivot_table(index='block_x', columns='block_y', values='accuracy', aggfunc='median')


# In[ ]:


bench_grid = bench_shared_df.query('time_optimized_ms>0.0').pivot_table(index='block_x', columns='block_y', values='fps_opt', aggfunc='median')
sns.heatmap(bench_grid, fmt='2.0f', annot=True, cmap='viridis')
bench_grid


# In[ ]:




