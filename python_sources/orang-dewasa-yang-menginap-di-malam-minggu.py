#!/usr/bin/env python
# coding: utf-8

# ## Menghitung Jumlah Orang Dewasa yang Menginap di Malam Minggu

# Import Library Python

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#  #### Melihat data dari `hotel_booking.csv`

# In[ ]:


dataHotel = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
dataHotel.head()


# #### Melihat semua kolom tabel yang tersedia

# In[ ]:


dataHotel.columns


# #### Membuat tabel baru dari beberapa kolom yang dikelompokkan berdasarkan jumlah `stays_in_weekend_nights`

# In[ ]:


jumlahDewasaWeekend=dataHotel.groupby(['adults','is_repeated_guest','arrival_date_month','arrival_date_year']).stays_in_weekend_nights.sum().reset_index()
jumlahDewasaWeekend.columns=['adults','is_repeated_guest','arrival_date_month','arrival_date_year','stays_in_weekend_nights']
jumlahDewasaWeekend


# #### Memilih orang dewasa yang merupakan *customer* baru

# In[ ]:


jumlahDewasaWeekend=jumlahDewasaWeekend[jumlahDewasaWeekend.is_repeated_guest != 1]
jumlahDewasaWeekend=jumlahDewasaWeekend[jumlahDewasaWeekend.adults != 0]
jumlahDewasaWeekend


# #### Menambah kolom baru `time_arrival` untuk menampung `arrival_date_month` dan `arrival_date_year`

# In[ ]:


jumlahDewasaWeekend['time_arrival'] = pd.to_datetime(jumlahDewasaWeekend.arrival_date_year.astype(str) + '-' + jumlahDewasaWeekend.arrival_date_month.astype(str))
jumlahDewasaWeekend=jumlahDewasaWeekend.sort_values(by=['time_arrival'])
jumlahDewasaWeekend


# #### Membuat tabel baru yang datanya dikelompokkan berdasarkan `stays_in_weekend_nights`

# In[ ]:


waktuKedatanganWeekend=jumlahDewasaWeekend.groupby(['time_arrival','adults']).stays_in_weekend_nights.sum().reset_index()
waktuKedatanganWeekend.columns=['time_arrival','adults','stays_in_weekend_nights']
waktuKedatanganWeekend


# Menambah kolom `year` dan`month` sehingga data dapat diurutkan berdasarkan tahun dan bulan

# In[ ]:


waktuKedatanganWeekend['year']=pd.DatetimeIndex(waktuKedatanganWeekend['time_arrival']).year
waktuKedatanganWeekend['month']=pd.DatetimeIndex(waktuKedatanganWeekend['time_arrival']).month
waktuKedatanganWeekend


# #### Visualisasi data menggunakan Seaborn

# In[ ]:


warnaPalet=sns.color_palette("cubehelix", 13)
grafik = sns.relplot(x='month',y='stays_in_weekend_nights', data=waktuKedatanganWeekend,col='year',kind='line',hue='adults', palette=warnaPalet)
grafik.fig.suptitle('Jumlah Dewasa Baru yang menginap di Malam Minggu setiap Bulan selama 3 Tahun', y=1.10)
grafik.set(xlabel="Bulan", ylabel="Menginap di Malam Minggu")
plt.show()


# #### Menghitung jumlah terbanyak grup `adults`

# In[ ]:


hitungDewasa = waktuKedatanganWeekend.groupby('adults').stays_in_weekend_nights.sum().reset_index()
hitungDewasa.columns = ['adults','stays_in_weekend_nights']
hitungDewasa


# #### Dari grafik di atas, perhitungan data dimulai dari bulan ke-7 2015 hingga bulan ke-8 2017. Jumlah orang dewasa yang memesan kamar di malam minggu, bervariasi mulai dari individual hingga rombongan 55 orang.Pada garis teratas di grafik, terdapat jumlah orang terbanyak yang datang, yaitu grup dari 2 orang sebanyak 87.409 kunjungan.
