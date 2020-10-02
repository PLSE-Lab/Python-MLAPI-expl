#!/usr/bin/env python
# coding: utf-8

# # Probabilistic Neural Networks: SimpleOCR
# 
# [Hendy Irawan](https://hendyirawan.com/), Februari 2018
# 
# Sumber: [Slides Pak Suyanto](https://www.dropbox.com/s/4p7frktuv5l5c0o/ML%2004%20ANN-MLP%20%26%20PNN.pptx?dl=0), [Slides Pak Anditya](https://www.dropbox.com/s/r7n1zjze3wkza2z/06%20-%20Probabilistic%20Neural%20Network.pptx?dl=0), [Video XoaX.net](https://www.dropbox.com/s/xrx2vz64v62mn2l/04b%20Probabilisitic%20Neural%20Networks.mp4?dl=0)
# 
# 1. **Probabilistic Neural Networks: SimpleOCR** (this one)
# 2. [Probabilistic Neural Networks: Smoothing Effect](https://www.kaggle.com/hendyirawan/probabilistic-neural-networks-smoothing-effect)
# 
# Contoh kasus penggunaan **Probabilistic Neural Networks (PNN)** adalah **SimpleOCR**:
# 
# ![SimpleOCR](https://machine-learning-course.github.io/syllabus/ann/simpleocr1.jpg)
# 
# Data _training_ tersedia di file `simpleocr.csv`.
# Gunakan `pandas` untuk mengambilnya.

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv('../input/simpleocr.csv')
train


# ## Arsitektur PNN
# 
# Dari data tersebut kita dapat menentukan arsitektur PNN yang digunakan:
# 
# * Cacah kelas $|C| = 3$ yaitu $C = \{\mathrm{BLUE}, \mathrm{RED}, \mathrm{GREEN\}}$
# * Cacah fitur: 2 yaitu *length* ($x_0$) dan *area* ($x_1$)
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

# ## Pengantar Fungsi Gaussian
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


# Yuk kita coba plot 3D:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sigma = 0.1
w_j = (0.5, 0.7)

# Plot source code from: https://stackoverflow.com/a/9170879/122441
x_0_range = np.linspace(w_j[0] - 3*sigma, w_j[0] + 3*sigma, 100)
x_1_range = np.linspace(w_j[1] - 3*sigma, w_j[1] + 3*sigma, 100)
X_0, X_1 = np.meshgrid(x_0_range, x_1_range)
fs = np.array( [gaussian_pdf2((x_0, x_1), sigma, w_j)
                for x_0, x_1 in zip(np.ravel(X_0), np.ravel(X_1))] )
FS = fs.reshape(X_0.shape)

fig = plt.figure('Fungsi Gaussian dengan 2 variabel')
ax = fig.add_subplot(111, projection='3d')
ax.set_title('$\sigma = %s, w_j = (%s, %s)$' % (sigma, w_j[0], w_j[1]))
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$f(x_0, x_1; \sigma, w_j)$')
ax.plot_surface(X_0, X_1, FS)
plt.show()


# Kita dapat mencoba memasukkan nilai tertentu, misalnya $x_0 = 0.2; x_1 = 0.6$ seperti pada contoh di video, dan hasilnya:

# In[ ]:


gaussian_pdf2(x = (0.2, 0.6),
              sigma = 0.1,
              w_j = [0.5, 0.7])


# In[ ]:


gaussian_pdf2(x = (0.2, 0.6),
              sigma = 0.1,
              w_j = [0.2, 0.5])


# Bandingkan hasil tadi dengan contoh di video: (dua neuron pattern berwarna biru)
#     
# ![SimpleOCR](https://machine-learning-course.github.io/syllabus/ann/simpleocr3.jpg)

# Untuk menghitung nilai fungsi Gaussian untuk semua neuron di _pattern layer_, sebelumnya kita dapat membuat matriks $W$ berisi fitur-fitur yang digunakan yaitu _length_ dan _area_.

# In[ ]:


import numpy as np

W = np.array([ (train['length'][d], train['area'][d]) 
              for d in range(len(train))])
W


# Lalu kita panggil fungsi `gaussian_pdf()` kita menggunakan matriks $W$ tadi.
# 
# $$\large
# \begin{align}
# \sigma &= 0.1 \\
# x &= (0.2, 0.6) \\
# p_j &= \mathrm{gaussian\_pdf2}(\mathbf{x}; \sigma, \mathbf{w}_j); j = [0 \ldotp\ldotp |D|-1]; \mathbf{w}_j \in W
# \end{align}
# $$

# In[ ]:


sigma = 0.1
x = (0.2, 0.6)
patterns = np.array([ gaussian_pdf2(x, sigma, w_j) for w_j in W ])
patterns


# Nilai category neurons kita dapatkan dari penjumlahan layer sejumlahnya, untuk masing-masing kelas.
# 
# $$\begin{align}
# c_{blue} &= p_0 + p_1 \\
# c_{red} &= p_2 + p_3 \\
# c_{green} &= p_5 + p_5 + p_6
# \end{align}$$

# In[ ]:


# Penjumlahan secara manual
c_blue = patterns[0] + patterns[1]
c_red = patterns[2] + patterns[3]
c_green = patterns[4] + patterns[5] + patterns[6]

print('c_blue = %s' % c_blue)
print('c_red = %s' % c_red)
print('c_green = %s' % c_green)


# Atau dapat kita menggunakan cara yang lebih fleksibel seperti ini:
# 
# $$ c_{k} = \sum{p_i}, i \in \{label(\mathbf{d_i}) = C_k\} $$

# In[ ]:


# Penjumlahan secara umum
c_blue = np.sum(
    (patterns[d] if train['label'][d] == 'BLUE' else 0)
    for d in range(len(train)) )
c_red = np.sum(
    (patterns[d] if train['label'][d] == 'RED' else 0)
    for d in range(len(train)) )
c_green = np.sum(
    (patterns[d] if train['label'][d] == 'GREEN' else 0)
    for d in range(len(train)) )
print('c_blue = %s' % c_blue)
print('c_red = %s' % c_red)
print('c_green = %s' % c_green)


# Bandingkan hasilnya dengan video: (yang dilingkari putih)
#     
# ![SimpleOCR](https://machine-learning-course.github.io/syllabus/ann/simpleocr4.jpg)
# 
# Dari sini, kita tinggal membandingkan saja.

# In[ ]:


categories = [('BLUE', c_blue),
              ('RED', c_red),
              ('GREEN', c_green)]
categories


# In[ ]:


best_label = None
best_value = None
for cat in categories:
    if not best_label or cat[1] > best_value:
        best_label = cat[0]
        best_value = cat[1]
        
print('Best label: %s. Best value: %s.' % (best_label, best_value))

