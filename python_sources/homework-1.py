#!/usr/bin/env python
# coding: utf-8

# **Quick tasks**
# 
# Pretty straightforward.

# In[ ]:


#Buat conditional dengan: indeks A untuk nilai >80, AB untuk 71-80, B untuk 61-70, dan C untuk <70
#print hasilnya
nilai = ___

#gunakan try-except
___


# In[ ]:


#an example of tuple
#like a list, but immutable (non editable)
#allows us to return multiple value
def powernum(val):
    quad = val ** 2
    cube = val ** 3
    return (quad, cube)

#try using powernum() and assign quad to a, cube to b, then print(a) and print(b)
#explain in markdown what happened.


# In[ ]:


import pandas as pd


# In[ ]:


baseball = pd.read_csv('../input/baseball.csv')
brics = pd.read_csv('../input/brics.csv')
iris = pd.read_csv('../input/iris.csv')


# > **Tasks**
# 1. Dari dataset brics, rapikan seperti saat latihan dan buat histogram. Lakukan analisis sebanyak mungkin dari yang sudah dipelajari. Silakan copy paste kode yang sudah kamu ketik pada latihan sebelumnya.
# 2. Dari dataset baseball, eksplorasi data dengan menggunakan pandas. Eksperimen sebanyak mungkin.
# 3. Dari dataset iris, lakukan plotting sebanyak mungkin dengan tetap memperhatikan keterbacaan.
# 4. Jelaskan (pada markdown) tentang subplots, kegunaannya, dan cara menggunakannya.
