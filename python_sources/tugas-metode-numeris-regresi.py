#!/usr/bin/env python
# coding: utf-8

# # Inisialisasi Awal

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

smooth = 20


# # Deskripsi Soal
# Diberikan data sebagai berikut

# | x | y |
# |------|------|
# | 1 | 0.4 |
# | 2 | 0.7 |
# | 2.5 | 0.8 |
# | 4 | 1.0 |
# | 6 | 1.2 |
# | 8 | 1.3 |
# | 8.5 | 1.4 |

# 1. Hitung persamaan garis (linear) yang mewakilinya dan koefisien korelasinya
# 2. Hitung persamaan berpangkat yang mewakilinya dan koefisien korelasinya
# 3. Hitung persamaan eksponensial yang mewakilinya dan hitung koefisien korelasinya
# 4. Hitung persamaan polinomial (orde-3) yang mewakilinya dan hitung koefisien korelasinya

# # Input Data

# In[ ]:


x = np.array([1.0, 2.0, 2.5, 4.0, 6.0, 8.0, 8.5])
y = np.array([0.4, 0.7, 0.8, 1.0, 1.2, 1.3, 1.4])


# ## Scatter Plot Data

# In[ ]:


plt.figure(figsize=(12,9))
plt.scatter(x, y, c='#0000FF')
plt.grid()
plt.xlabel('x')
plt.ylabel('y') 
plt.show()


# # Regresi Linear (LSE)
# ## Perhitungan
# Kita mencari persamaan linear $$y = a + bx$$
# dengan menggunakan jumlah kuadrat eror ($D$) $$D = \sum_{i=1}^n E_i = \sum_{i=1}^n (y_i - a - bx_i)^2$$
# agar nilai $D$ minimum maka kita turunkan persamaan $D$ terhadap paramater $a$ dan $b$ dan kemudian disamadengankan nol sehingga kita mendapat persamaan $a$ dan $b$ sebagai berikut
# $$a = \overline{y} - b\overline{x}$$
# $$ $$
# $$b = \frac{n\sum_{i=1}^nx_iy_i-\sum_{i=1}^nx_i\sum_{i=1}^ny_i}{n\sum_{i=1}^nx_i^2-(\sum_{i=1}^nx_i)^2}$$

# In[ ]:


n = len(x)
def regresiLinear(xr,yr):
    xryr = xr*yr
    xr2 = xr*xr
    br = (n*xryr.sum()-xr.sum()*yr.sum())/(n*xr2.sum()-xr.sum()**2)
    ar = (np.mean(yr) - br*np.mean(xr))
    return ar, br

a1, b1 = regresiLinear(x,y)
print("a = {}\nb = {}".format(a1,b1))


# ## Persamaan Regresi

# In[ ]:


s1 = "y = {:.7f} + {:.7f}x".format(a1, b1)
print("Persamaan Linear (LSE): y = {}".format(s1))


# ## Metrik Evaluasi
# Ada beberapa matriks evaluasi yang dapat digunakan untuk mengevaluasi model yang telah didapat. 

# ### Mean Squared Error (MSE)
# Matriks evaluasi yang pertama adalah Mean Squared Error (MSE). MSE dirumuskan dengan
# $$MSE = \frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2 = \frac{1}{n}\sum_{i=1}^n(y_i-a-bx_i)^2$$

# In[ ]:


def eval1(l):
    return (a1+b1*l)

def MSE(yPredr):
    yDtr = (y-y.mean())**2
    yDr = (y - yPredr)**2
    return (yDr.sum()/n)

yPred1 = eval1(x)
modelMSE1 = MSE(yPred1)
baseMSE = MSE(y.mean())
print("Mean Squared Error (MSE) = {}".format(modelMSE1))


# ### Root Mean Squared Error (RMSE)
# Matriks evaluasi yang kedua adalah Root Mean Squared Error (RMSE). RMSE dirumuskan dengan
# $$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2} = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-a-bx_i)^2}$$

# In[ ]:


def RMSE(r):
    return np.sqrt(r)

modelRMSE1 = RMSE(modelMSE1)
print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE1))


# ### Mean Absolute Error (MAE)
# Matriks evaluasi yang ketiga adalah Mean Absolute Error (MAE). MAE dirumuskan dengan
# $$MSE = \frac{1}{n}\sum_{i=1}^n|y_i-\hat{y_i}| = \frac{1}{n}\sum_{i=1}^n|y_i-a-bx_i|$$

# In[ ]:


def MAE(yPredr):
    dAbsr = np.absolute(y - yPredr)
    return (dAbsr.sum()/n)

modelMAE1 = MAE(yPred1)
print("Mean Absolute Error (MAE)) = {}".format(modelMAE1))


# ### Koefisien Korelasi ($r$) dan Derajat Kesesuaian ($R$)
# Matriks evaluasi yang terakhir adalah derajat kesesuaian, untuk mengukur derajat kesesuaian dari persamaan yang didapat, dihitung nilai koefisien korelasi $r$ dan derajat kesesuaian $R$:
# $$R = r^2$$
# $$r = \sqrt{\frac{D_t-D}{D_t}} = \sqrt{1-\frac{MSE(model)}{MSE(baseline)}}$$
# dengan
# $$D_t = MSE(baseline)n = n\frac{1}{n}\sum_{i=1}^n(y_i-\bar{y})^2 = \sum_{i=1}^n(y_i-\bar{y})^2$$
# $$D = MSE(model)n = n\frac{1}{n}\sum_{i=1}^n(y_i-a-bx_i)^2 = \sum_{i=1}^n(y_i-a-bx_i)^2$$

# In[ ]:


def rR(MSEmr):
    Rr = (1 - (MSEmr/baseMSE))
    rr = np.sqrt(Rr)
    return rr, Rr

r1, R1 = rR(modelMSE1)
print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r1, R1))


# ## Plot Garis Regresi

# In[ ]:


def plot(s,f):
    xreg = np.array(np.arange(int(x[0]-1)*smooth,int(x[-1]+1)*smooth))
    xreg = xreg/smooth
    print("x regresi: {}".format(xreg))
    yreg = (f(xreg))
    print("y regresi: {}".format(yreg))
    plt.figure(figsize=(12,9))
    plt.scatter(x, y, c="#0000FF", label="Data")
    plt.plot(xreg, yreg, "r", label=s)
    plt.legend(loc="lower right")
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    
plot(s1, eval1)


# # Regresi Nonlinear (Persamaan Berpangkat)
# ## Perhitungan
# Kita mencari persamaan nonlinear berpangkat $$y = ax^b$$
# Untuk menyelesaikannya, kita dapat melinearkan persamaan tersebut menjadi $$\log{y} = b\log{x}+\log{a}$$
# Selanjutnya kita dapat menggunakan metode regresi linear (LSE) untuk menyelesaikan persamaan diatas dengan $p = \log{x}$ dan $q = \log{y}$ sehingga menjadi
# $$ q = A + Bp$$

# In[ ]:


p = np.log(x)
q = np.log(y)

A2, B2 = regresiLinear(p,q)
print("A = {}\nB = {}".format(A2, B2))


# ## Persamaan Regresi

# In[ ]:


a2 = np.exp(A2)
b2 = B2
s2 = "{:.7f}x^({:.7f})".format(a2,b2)
print("Persamaan Berpangkat: y = {}".format(s2))


# ## Metrik Evaluasi

# ### Mean Squared Error (MSE)

# In[ ]:


def eval2(l):
    return (a2*(l**b2))

yPred2 = eval2(x)
modelMSE2 = MSE(yPred2)
print("Mean Squared Error (MSE) = {}".format(modelMSE2))


# ### Root Mean Squared Error (RMSE)

# In[ ]:


modelRMSE2 = RMSE(modelMSE2)
print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE2))


# ### Mean Absolute Error (MAE)

# In[ ]:


modelMAE2 = MAE(yPred2)
print("Mean Absolute Error (MAE) = {}".format(modelMAE2))


# ### Koefisien Korelasi ($r$) dan Derajat Kesesuaian ($R$)

# In[ ]:


r2, R2 = rR(modelMSE2)
print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r2, R2))


# ## Plot Garis Regresi

# In[ ]:


plot(s2, eval2)


# # Regresi Nonlinear (Persamaan Eksponensial)
# ## Perhitungan
# Kita mencari persamaan nonlinear eksponensial $$y = ae^{bx}$$
# Untuk menyelesaikannya, kita dapat melinearkan persamaan tersebut menjadi $$\ln{y} = bx\ln{e}+\ln{a}$$
# Selanjutnya kita dapat menggunakan metode regresi linear (LSE) untuk menyelesaikan persamaan diatas dengan $v = x$ dan $w = \ln{y}$ sehingga menjadi
# $$ w = A + Bv$$

# In[ ]:


v = x
w = np.log(y)

A3, B3 = regresiLinear(v, w)
print("A = {}\nB = {}".format(A3, B3))


# ## Persamaan Regresi

# In[ ]:


a3 = np.exp(A3)
b3 = B3
s3 = "{:.7f}e^({:.7f}x)".format(a3,b3)
print("Persamaan Eksponensial: y = {}".format(s3))


# ## Metrik Evaluasi

# ### Mean Squared Error (MSE)

# In[ ]:


def eval3(l):
    return (a3 * (np.e**(l*b3)))

yPred3 = eval3(x)
modelMSE3 = MSE(yPred3)
print("Mean Squared Error (MSE) = {}".format(modelMSE3))


# ### Root Mean Squared Error (RMSE)

# In[ ]:


modelRMSE3 = RMSE(modelMSE3)
print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE3))


# ### Mean Absolute Error (MAE)

# In[ ]:


modelMAE3 = MAE(yPred3)
print("Mean Absolute Error (MAE) = {}".format(modelMAE3))


# ### Koefisien Korelasi ($r$) dan Derajat Kesesuaian ($R$)

# In[ ]:


r3, R3 = rR(modelMSE3)
print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r3, R3))


# ## Plot Garis Regresi

# In[ ]:


plot(s3, eval3)


# # Regresi Nonlinear (Persamaan Polinomial Orde-3)
# ## Perhitungan
# Persamaan nonlinear polinimial orde-r $$y = a_o + a_1x + a_2x^2 + \dots + a_rx^r$$
# dapat kita cari dengan menggunakan jumlah kuadrat eror ($D$) 
# $$D = \sum_{i=1}^n E_i = \sum_{i=1}^n (y_i - (a_o + a_1x + a_2x^2 + \dots + a_rx^r))^2$$
# agar nilai $D$ minimum maka kita turunkan persamaan $D$ terhadap tiap-tiap koefisien dari polinomial dan kemudian disamadengankan nol sehingga
# $$\frac{\partial{D}}{\partial{a_0}} = -2\sum_{i=1}^n(y_i - (a_o + a_1x_i + a_2x_i^2 + \dots + a_rx_i^r)) = 0$$
# $$\frac{\partial{D}}{\partial{a_1}} = -2\sum_{i=1}^{n}x_i(y_i - (a_o + a_1x_i + a_2x_i^2 + \dots + a_rx_i^r)) = 0$$
# $$\frac{\partial{D}}{\partial{a_2}} = -2\sum_{i=1}^{n}x_i^2(y_i - (a_o + a_1x_i + a_2x_i^2 + \dots + a_rx_i^r)) = 0$$
# $$\vdots$$
# $$\frac{\partial{D}}{\partial{a_r}} = -2\sum_{i=1}^{n}x_i^r(y_i - (a_o + a_1x_i + a_2x_i^2 + \dots + a_rx_i^r)) = 0$$
# Dapat kita lihat bahwa persamaan diatas dapat kita selesaikan dengan menggunakan metode persamaan linear $AX = B$ dengan
# $$\begin{gather}  \begin{bmatrix} n & \sum{x_i} & \sum{x_i^2} & \cdots & \sum{x_i^r} \\ \sum{x_i} & \sum{x_i^2} & \sum{x_i^3} & \cdots & \sum{x_i^{r+1}} \\ \sum{x_i^2} & \sum{x_i^3} & \sum{x_i^4} & \cdots & \sum{x_i^{r+2}} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \sum{x_i^r} & \sum{x_i^{r+1}} & \sum{x_i^{r+2}} & \cdots & \sum{x_i^{r+r}} \\ \end{bmatrix} \begin{bmatrix} a_0 \\ a_1 \\ a_2 \\ \vdots \\ a_r \end{bmatrix} = \begin{bmatrix} \sum{y_i} \\ \sum{x_iy_i} \\ \sum{x_i^2y_i} \\ \vdots \\ \sum{x_i^ry_i} \end{bmatrix}  \end{gather}$$

# ### Penyelesaian Sistem Persamaan Linear $AX = B$

# Dalam hal ini kita ingin mencari persamaan polinimial orde-3. Untuk itu kita membutuhkan matriks $A = (a_{ij}) \in \mathbb{R}^{4x4}$, $X = (x_{ij}) \in \mathbb{R}^{4x1}$, dan $B = (b_{ij}) \in \mathbb{R}^{4x1}$
# $$
# A = 
#     \begin{bmatrix}
#     n & \sum{x_i} & \sum{x_i^2} & \sum{x_i^3} \\
#     \sum{x_i} & \sum{x_i^2} & \sum{x_i^3} & \sum{x_i^{4}} \\
#     \sum{x_i^2} & \sum{x_i^3} & \sum{x_i^4} & \sum{x_i^{5}} \\
#     \sum{x_i^3} & \sum{x_i^4} & \sum{x_i^5} & \sum{x_i^{6}} \\
#     \end{bmatrix}
# $$
# $$ $$
# $$
# X = 
#     \begin{bmatrix}
#     a_0 \\
#     a_1 \\
#     a_2 \\
#     a_3 \\
#     \end{bmatrix}
# $$
# $$ $$
# $$
# B = 
#     \begin{bmatrix}
#     \sum{y_i} \\
#     \sum{x_iy_i} \\
#     \sum{x_i^2y_i} \\
#     \sum{x_i^3y_i}
#     \end{bmatrix}
# $$
# Sedemikian rupa sehingga 
# $$ AX = B $$

# In[ ]:


orde = 3

def ai(i):
    return [np.sum(x**j) for j in range(i,i+orde+1)]

A4 = np.matrix([ai(i) for i in range(0,orde+1)])
print("Matriks A:")
print(A4)


# In[ ]:


B4 = np.array([np.sum((x**i)*y) for i in range(0,orde+1)])
print("Matriks B:")
print(B4)


# Matrik $X$ dapat kita cari dengan $$X = A^{-1}B$$

# In[ ]:


X4 = np.linalg.inv(A4).dot(B4)
print("Didapatkan Matriks X:")
print(X4)


# ## Persamaan Regresi

# In[ ]:


print("Persamaan Polinomial orde-{}: y = ".format(orde), end="")
s4 = ""
for i,j in enumerate(np.nditer(X4)):
    if(i==0):
        now = "{:.7f}".format(j) 
        print(now, end="")
        s4 += now
    else:
        now = " + {:.7f}x^{}".format(j,i)
        print(now, end="")
        s4 += now


# ## Metrik Evaluasi

# ### Mean Squared Error (MSE)

# In[ ]:


def eval4(l, coeff):
    result = coeff[-1]
    for i in range(-2, -len(coeff)-1, -1):
        result = result*l + coeff[i]
    return result

yPred4 = np.array([eval4(i,X4.tolist()[0]) for i in x.tolist()])
modelMSE4 = MSE(yPred4)
print("Mean Squared Error (MSE) = {}".format(modelMSE4))


# ### Root Mean Squared Error (RMSE)

# In[ ]:


modelRMSE4 = RMSE(modelMSE4)
print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE4))


# ### Mean Absolute Error (MAE)

# In[ ]:


modelMAE4 = MAE(yPred4)
print("Mean Absolute Error (MAE) = {}".format(modelMAE4))


# ### Koefisien Korelasi ($r$) dan Derajat Kesesuaian ($R$)

# In[ ]:


r4, R4 = rR(modelMSE4)
print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r4, R4))


# ## Plot Garis Regresi

# In[ ]:


xreg4 = np.array(range(int(x[0]-1)*smooth,int(x[-1]+1)*smooth))
xreg4 = xreg4/smooth
print("x regresi: {}".format(xreg4))
yreg4 = np.array([eval4(i,X4.tolist()[0]) for i in xreg4.tolist()])
print("y regresi: {}".format(yreg4))


# In[ ]:


plt.figure(figsize=(12,9))
plt.scatter(x, y, c="#0000FF", label="Data")
plt.plot(xreg4, yreg4, "r", label="y = " + s4)
plt.legend(loc="lower right")
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

