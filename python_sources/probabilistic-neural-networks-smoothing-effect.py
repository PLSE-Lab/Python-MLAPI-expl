#!/usr/bin/env python
# coding: utf-8

# # Probabilistic Neural Networks: Smoothing Effect
# 
# [Hendy Irawan](https://hendyirawan.com/), Februari 2018
# 
# Sumber: [Slides Pak Suyanto](https://www.dropbox.com/s/4p7frktuv5l5c0o/ML%2004%20ANN-MLP%20%26%20PNN.pptx?dl=0), [Slides Pak Anditya](https://www.dropbox.com/s/r7n1zjze3wkza2z/06%20-%20Probabilistic%20Neural%20Network.pptx?dl=0), [Video XoaX.net](https://www.dropbox.com/s/xrx2vz64v62mn2l/04b%20Probabilisitic%20Neural%20Networks.mp4?dl=0)
# 
# 1. [Probabilistic Neural Networks: SimpleOCR](https://www.kaggle.com/hendyirawan/probabilistic-neural-networks-simpleocr)
# 2. **Probabilistic Neural Networks: Smoothing Effect** (this one)

# Nature of the PDF varies as we change $\sigma$ (_standard deviation_ = _smoothing parameter_).
# 
# * Small $\sigma$ creates distinct modes
# * Larger $\sigma$ allows interpolation between points
# * Very Large $\sigma$ approximate PDF to Gaussian
# 
# Fungsi Gaussian adalah _probability density function_ (PDF) yang umum digunakan untuk PNN. Fungsi Gaussian untuk satu variabel $x$ sebagai berikut:
# 
# $$\large f(x; \sigma, w) = e^{-\frac{1}{2}(\frac{x - w}{\sigma})^2} = e^{-\frac{(x - w)^2}{2\sigma^2}} $$
# 
# dengan $w$ adalah nilai yang diinginkan atau pusat puncak, dan konstanta $\sigma$ (simpangan baku atau disebut juga _Gaussian RMS width_) merepresentasikan "kelebaran" dari bentuk "kurva lonceng" yang dihasilkan.
# 
# Dalam Python:

# In[ ]:


import math

def gaussian_pdf1(x, sigma, w):
    return math.exp( -(x - w)**2 / (2 * sigma**2) )


# Kita dapat memplot fungsi Gaussian tersebut:    

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
# If you get "ImportError: DLL load failed: The specified procedure could not be found."
# see https://github.com/matplotlib/matplotlib/issues/10277#issuecomment-366136451
# Short answer: pip uninstall cntk
import matplotlib.pyplot as plt

sigma = 0.1
w = 0.5
x = np.linspace(w - 2, w + 2, 100)
fig = plt.figure('Fungsi Gaussian')
ax = fig.add_subplot(111)
ax.set_title('Fungsi Gaussian dengan $\sigma = %s, w = %s$' % (sigma, w))
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x; \sigma, w)$')
ax.grid(which='major')
ax.plot(x, [gaussian_pdf1(_, sigma, w) for _ in x])
plt.show()


# Bagaimana efek konstanta $\sigma$ (simpangan baku) pada fungsi Gaussian yang dihasilkan?
# 
# Yuk kita coba.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import numpy as np
# If you get "ImportError: DLL load failed: The specified procedure could not be found."
# see https://github.com/matplotlib/matplotlib/issues/10277#issuecomment-366136451
# Short answer: pip uninstall cntk
import matplotlib.pyplot as plt

w = 0.5
x = np.linspace(w - 2, w + 2, 100)
fig = plt.figure('Fungsi Gaussian')
ax = fig.add_subplot(111)
ax.set_title('Fungsi Gaussian dengan $\sigma = \{0.1, 0.2, 0.5, 1.0\}; w = %s$' % (w))
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x; \sigma, w)$')
ax.grid(which='major')
ax.plot(x, [gaussian_pdf1(_, 0.1, w) for _ in x], label='$\sigma = 0.1$')
ax.plot(x, [gaussian_pdf1(_, 0.2, w) for _ in x], label='$\sigma = 0.2$')
ax.plot(x, [gaussian_pdf1(_, 0.5, w) for _ in x], label='$\sigma = 0.5$')
ax.plot(x, [gaussian_pdf1(_, 1.0, w) for _ in x], label='$\sigma = 1.0$')
plt.legend()
plt.show()


# Contoh kasus **Probabilistic Neural Networks (PNN)** yang kita gunakan adalah **SimpleOCR**:
# 
# ![SimpleOCR](https://machine-learning-course.github.io/syllabus/ann/simpleocr1.jpg)
# 
# Data _training_ tersedia di file `simpleocr.csv`.
# Gunakan `pandas` untuk mengambilnya.

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/simpleocr.csv')
train


# ## Arsitektur PNN
# 
# Dari data tersebut kita dapat menentukan arsitektur PNN yang digunakan:
# 
# * Cacah kelas: $|C| = 3$ yaitu $C = \{\mathrm{BLUE}, \mathrm{RED}, \mathrm{GREEN\}}$
# * Cacah fitur: $|X| = 2$ yaitu *length* ($x_0$) dan *area* ($x_1$)
# * Cacah sampel $|D| = n = 7$, yaitu $D = (\mathbf{d_0}, \ldots, \mathbf{d_6})$
# * Matriks fitur $W$ berukuran $7 \times 2$, yaitu $W = (\mathbf{w_0}, \ldots, \mathbf{w_6})$
# 
# Arsitektur PNN tersebut menggunakan _hyperparameters_ berikut:
# 
# * _Input layer_ sebanyak 2 neuron, untuk _length_ dan _area_
# * _Pattern layer_ sebanyak 7 neuron, 2 untuk BLUE, 2 untuk RED, dan 3 untuk GREEN
# * _Category layer_ sebanyak 3 neuron, untuk masing-masing label BLUE, RED, dan GREEN
# * _Output_ memilih neuron dari _category_ layer dengan nilai terbesar sebagai hasil prediksi
# 
# ![SimpleOCR](https://machine-learning-course.github.io/syllabus/ann/simpleocr2.jpg)

# ## Pattern Layer
# 
# Fungsi densitas peluang (PDF) yang umum digunakan pada _pattern layer_ adalah Gaussian, dengan konstanta $\sigma$ tertentu. Untuk masing-masing neuron $\mathbf{w_j}; i = 0\ldotp\ldotp|D|-1; |D| = 7$ di _pattern layer_, maka fungsi Gaussian dengan 2 variabel $\mathbf{x} = (x_0, x_1)$ menjadi:
# 
# $$\large
# \begin{align}
# f(x_0, x_1; \sigma, \mathbf{w_j}) &= e^{-\frac{\left\Vert\mathbf{x} - \mathbf{w_j}\right\Vert^2}{2 \sigma^2}} \\
# &= e^{-\frac{\left(x_0 - w_{j,0}\right)^2 + \left(x_1 - w_{j,1}\right)^2}{2 \sigma^2}}
# \end{align}
# $$
# 
# (Ini fungsi Gaussian yang **sudah disederhanakan**, dengan asumsi PDF yang digunakan untuk semua kelas adalah sama, **dengan parameter $\sigma$ yang sama**.)
# 
# Fungsi tersebut dapat kita tulis di Python sebagai berikut:

# In[ ]:


import math

def gaussian_pdf2(x, sigma, w_j):
    return math.exp(
        -( (x[0] - w_j[0])**2 + (x[1] - w_j[1])**2 ) /
        (2 * sigma**2) )


# ## Mencari $\sigma_k$
# 
# $\sigma_k$ artinya setiap kelas $C_k \in C$ memiliki nilai _smoothing parameter_ sendiri-sendiri (sehingga, fungsi Gaussian yang dihasilkan pun berbeda-beda tingkat "kelandaian"nya).
# 
# Bagaimana menentukan $\sigma_k$ ? Yuk kita lihat caranya.
# 
# ### Tahap pertama
# 
# * `For` setiap pola $w_j;\ j = 0 \ldotp\ldotp n-1$
# 
#   * Bentuk unit pola dengan memasukkan vektor bobot $w_j$
#   * Hubungkan unit pola pada unit penjumlah untuk kelas $C_k$ yang sesuai
# 
# * Tentukan konstanta $|C_k|$ untuk setiap unit penjumlah (berikut menggunakan notasi [Iverson bracket](https://math.stackexchange.com/questions/1468340/how-widely-known-are-iverson-brackets))
# 
# $$ |C_k| = \sum_{j=0}^{n-1} [ \mathrm{label}_j = C_k ] $$
# 
# ### Tahap kedua
# 
# * `For` setiap pola $w_j;\ j = 0 \ldotp\ldotp n-1$
# 
#   * $k$ : indeks kelas $w_j$
#   * Cari $d_j$ : jarak dengan pola terdekat lain pada kelas $k$.
#   
#     $$\begin{align}
# d_j &= \underset{i \in \{\mathrm{label}_i = C_k,\ i \neq j\}}{\operatorname{min}} \left\Vert w_j - w_i \right\Vert \\
#     &= \underset{i \in \{\mathrm{label}_i = C_k,\ i \neq j\}}{\operatorname{min}} \sqrt{ (w_{j,0} - w_{i,0})^2 + \cdots + (w_{j,|X|-1} - w_{i,|X|-1})^2 }
# \end{align}
# $$
# 
#     Bila tidak ada pola lain, maka $d_j = 1$.
#     
#   * $d_{tot}[k] = d_{tot}[k] + d_j$
# 
# * Tentukan $g$ (secara _brute force_)
# * `For` setiap kelas $k$
# 
#   * $d_{avg}[k] = \frac{d_{tot}[k]}{|C_k|}$
#   * $\sigma_k = g \cdot d_{avg}[k]$
# 

# In[ ]:


# Tahap pertama:
# For setiap pola w_j
#   Bentuk unit pola dengan memasukkan vektor bobot w_j
W = train[['length', 'area']].values
print('W = %s' % W)

#   Hubungkan unit pola pada unit penjumlah untuk kelas C_k yang sesuai
def sample_indexes_for_category(C_k):
    matching_samples_index = list(train[train.label == C_k].index)
    print('Samples untuk C_k=%s: %s' % (C_k, matching_samples_index))
    return matching_samples_index

C = train.label.unique()
print('Classes: %s' % C)

sample_indexes_for_category(C[0])
sample_indexes_for_category(C[1])
sample_indexes_for_category(C[2])


# * Tentukan konstanta $|C_k|$ untuk setiap unit penjumlah (berikut menggunakan notasi [Iverson bracket](https://math.stackexchange.com/questions/1468340/how-widely-known-are-iverson-brackets))
# 
# $$ |C_k| = \sum_{j=0}^{n-1} [ \mathrm{label}_j = C_k ] $$

# In[ ]:


# Tentukan konstanta |C_k| untuk setiap unit penjumlah
def count_label(C_k): return len(train[train.label == C_k])

C_count = [count_label(C_k) for C_k in C]
print('C_count: %s' % C_count)


# ### Tahap kedua
# 
# * `For` setiap pola $w_j;\ j = 0 \ldotp\ldotp n-1$
# 
#   * $k$ : indeks kelas $w_j$
#   * Cari $d_j$ : jarak dengan pola terdekat lain pada kelas $k$.
#   
#     $$\begin{align}
# d_j &= \underset{i \in \{\mathrm{label}_i = C_k,\ i \neq j\}}{\operatorname{min}} \left\Vert w_j - w_i \right\Vert \\
#     &= \underset{i \in \{\mathrm{label}_i = C_k,\ i \neq j\}}{\operatorname{min}} \sqrt{ (w_{j,0} - w_{i,0})^2 + \cdots + (w_{j,|X|-1} - w_{i,|X|-1})^2 }
# \end{align}
# $$
# 
#     Bila tidak ada pola lain, maka $d_j = 1$.

# In[ ]:


# Tahap kedua:
# For setiap pola w_j
def find_d_j(j):
    # k = indeks kelas w_j
    C_k = train.label[j]
    k = np.where(C==C_k)[0][0]
    # Cari d_j : jarak dengan pola terdekat lain pada kelas k
    sample_indexes = sample_indexes_for_category(C_k)
    sample_indexes.remove(j)
    print('For W[%s]: C_k=%s k=%s. Sample indexes (other): %s' % (j, C_k, k, sample_indexes))
    d_j_list = [np.linalg.norm(W[j] - W[sample_index]) for sample_index in sample_indexes]
    d_j = np.amin(d_j_list) or 1.0
    print('d_j list: %s => d_j = %s' % (d_j_list, d_j))
    return d_j

find_d_j(0)
find_d_j(1)
find_d_j(5)


#   * $d_{tot}[k] = d_{tot}[k] + d_j$

# In[ ]:


# d_tot[k] = d_tot[k] + d_j
def find_d_tot(C_k):
    return np.sum(find_d_j(j) for j in sample_indexes_for_category(C_k))

d_tot = np.array([find_d_tot(C_k) for C_k in C])
print('d_tot[0] = %s' % d_tot[0])
print('d_tot[1] = %s' % d_tot[1])
print('d_tot[2] = %s' % d_tot[2])


# * Tentukan $g$ (secara _brute force_)
# * `For` setiap kelas $k$
# 
#   * $d_{avg}[k] = \frac{d_{tot}[k]}{|C_k|}$
#   * $\sigma_k = g \cdot d_{avg}[k]$

# In[ ]:


# Tentukan g (brute force)
g = 2.0

# For setiap kelas k
#   d_avg[k] = d_tot[k] / |C_k|
d_avg = d_tot / C_count
print('d_avg = %s' % d_avg)

#   sigma_k = g . d_avg[k]
sigmas = g * d_avg
print('sigmas = %s' % sigmas)

