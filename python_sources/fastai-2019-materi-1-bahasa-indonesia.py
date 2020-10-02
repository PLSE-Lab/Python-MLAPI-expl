#!/usr/bin/env python
# coding: utf-8

# > The purpose of this notebook and series is to teach Indonesian people about fast.ai deep learning course speficically for those who can't understand English very well.

# # Materi 1 - Klasifikasi Binatang
# 
# Di materi ini kita akan membuat sebuah _classifier_ yang dapat mengklasifikasikan binatang. Kita akan menggunakan _library_ dari fast.ai  dan beberapa _library_ terkenal lainnya.
# 
# ```python
# %reload_ext autoreload
# %autoreload 2
# ```
# adalah _magic commands_ dari jupyter notebook. Bertujuan untuk melakukan _load_ modul sebelum dieksekusi. Lebih lanjutnya silahkan baca [ini](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html).
# 
# Perlu diperhatikan fast.ai tidak mengikuti PEP8 dalam materinya. Sehingga akan ada beberapa kode yang _anti pattern_. Salah satu contohnya ialah _import module_.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
from fastai.metrics import error_rate


# Gunakan batch size 64. Jika terjadi kasus GPU out of memory maka turunya batch size menjadi 32 lalu 16 lalu 8.

# In[ ]:


# bs: batch size
bs = 64


# Data yang digunakan ialah __Oxford-IIIT Pet Dataset__ yang terdiri dari 12 jenis kucing dan 25 jenis anjing. Dari paper mereka di tahun 2012, akurasi tertinggi dari model ialah 59.21%.

# In[ ]:


# download dataset
path = untar_data(URLs.PETS)
# lihat path
path


# In[ ]:


# lihat apa saja yang ada di path
path.ls()


# Dataset kita memiliki dua folder yaitu __annotations__ dan __images__. Pathlib python memiliki satu karakter khusus yaitu '/' untuk membuat sebuah path file. Silahkan kunjungi [ini untuk lebih jelasnya](https://docs.python.org/3/library/pathlib.html)

# In[ ]:


path_anno = path/'annotations'
path_img = path/'images'


# Fungsi `get_image_files` dari fastai bertujuan untuk mengambil image file.

# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]


# Nama file memiliki pola. Polanya yaitu jenis lalu diikuti garis bawah lalu nomor dan ekstensi file. Kita dapat mengekstrak/mengambil jenis anjing/kucing menggunakan _regular expression_ python. Selengkapnya [cek di sini](https://docs.python.org/3.6/library/re.html).
# 
# Nama jenis anjing/kucing ini selanjutnya kita namakan `label`.

# In[ ]:


# set random seed agar dapat diulangi kedepannya
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# Kelas `ImageDataBunch` merupakan sebuah kelas yang bertujuan untuk mengumpulkan dataset dan melakukan berbagai macam lainnya. Selengkapnya silahkan buka dokumentasi fastai. `from_name_re` digunakan karena kita ingin mengekstrak label dari dataset menggunakan _regular expression (regex)_.
# 
# Arti tiap parameternya sebagai berikut:
# - `path_img` merupakan path gambar
# - `fnames` merupakan kumpulan nama file
# - `pat` merupakan perintah regex kita
# - `ds_tfms` merupakan transformasi gambar
# - `size` merupakan ukuran input gambar
# - `bs` merupakan ukuran batch size
# 
# Dataset kemudian di normalisasi menggunakan nilai rataan dan standar deviasi dari imagenet.

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, 
                                   fnames, 
                                   pat, 
                                   ds_tfms=get_transforms(), 
                                   size=224, 
                                   bs=bs).normalize(imagenet_stats)


# Lihat dataset beserta labelnya dengan memanggil `show_batch`

# In[ ]:


data.show_batch(rows=3, figsize=(7, 6))


# In[ ]:


print(data.c) # banyaknya kelas
print(data.classes) # nama nama kelas


# Kita akan menggunakan sebuah pretrained model resnet34. Resnet43 memiliki 34 layer dan menggunakan _residual block_. Model ini telah dilatih menggunakan dataset imagenet yang dapat mengklasifikasikan 1000 kelas.

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/tmp/models')


# Perhatikan bahwa di Kaggle saya menambahkan parameter `model_dir`. Parameter tersebut bertujuan agar kita dapat menyimpan model. Kaggle sendiri tidak bisa menjadi tempat penyimpanan maka dari itu kita harus menggunakan parameter tersebut. 

# In[ ]:


learn.model


# Selanjutnya kita menggunakan __One Cycle Policy__ yang merupakan metode _hyperparameter tuning_ dengan merubah nilai _learning rate_ sehingga dapat melakukan lompatan jauh dan mencapai _flat local minima_. Selengkapnya [baca di sini](https://sgugger.github.io/the-1cycle-policy.html)

# In[ ]:


learn.fit_one_cycle(4) # 4 epoch


# Hanya dengan empat _epoch_ kita bisa memperoleh error yang sangat kecil sebesar 0.065 atau akurasi 1-0.065 = 93.5%. Selannjutnya save model.

# In[ ]:


# save model yang telah dilatih
learn.save('stage-1')


# Selanjutnya, mari melakukan interpretasi hasil latihan model.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


# plot gambar apa saja yang memiliki loss yang besar
interp.plot_top_losses(9, figsize=(15, 11))


# In[ ]:


# plot confusion matrixnya
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)


# In[ ]:


# lihat klasifikasi apa yang paling banyak yang salah
interp.most_confused(min_val=2)


# Sekarang mari kita unfreeze pretrained model kita. Sehingga dia akan belajar atau melakukan tuning sedikit agar kita dapat meningkatkan akurasi model kita.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# Wow! Akurasi model turun drastis. Error kita sekarang 0.12 yang awalnya 0.065. Hal ini terjadi karena kita melatih pretrained model dengan learning rate yang sama. Padahal untuk layer yang awal, kita tidak usah melakukan perubahan yang berarti. Untuk mengatasi hal ini, pertama mari kita load model kita yang telah jadi lalu mencari learning rate dan melatih ulang model kita menggunakan learning rate yang telah kita peroleh.

# In[ ]:


# load model
learn.load('stage-1')
learn.lr_find()


# In[ ]:


# lihat hasil lr finder
learn.recorder.plot()


# Dari grafik di atas, setelah _learning rate_ 1e-03 _loss_ langsung naik. Maka dari itu pilih _learning rate_ sebelum itu. Biasanya akan dipilih 10x lebih rendah dari batas sebagai titik akhir dan 100x dari titik akhir sebagai titik awal.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))


# Terlihat penurunan model kita yang awalnya 0.065 menjadi 0.066. Hal ini menandakan _unfreeze layer_ akan membuat model semakin bagus atau agak jelek sedikit. Ini diserahkan ke pembaca untuk melakukan eksperiment berulang kali.
# 
# Dari tadi kita menggunakan resnet34. Sekarang mari gunakan resnet50 yang memiliki 50 layer. Karena resnet50 memiliki 50 layer maka resnet50 akan memakan banyak memori. Maka dari itu, mari turunkan nilai _batch size_ dua kali lipat dari sebelumnya. Kita menggunakan dua metrics yaitu akurasi dan nilai eror.

# In[ ]:


data = ImageDataBunch.from_name_re(path_img, 
                                   fnames, 
                                   pat, 
                                   ds_tfms=get_transforms(), 
                                   size=299, 
                                   bs=bs//2).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir='/tmp/models')


# In[ ]:


# panggil lr finder
learn.lr_find()
learn.recorder.plot()


# Untuk saat ini kita tidak menggunakan `max_lr` karena kita tidak melakukan _unfreezing model_. 

# In[ ]:


learn.fit_one_cycle(8)


# Wow! Akurasi kita meningkat hingga 93.7%. Sekarang mari lakukan _unfreeze model_. 

# In[ ]:


learn.save('stage-1-50')
learn.unfreeze()


# Lakukan _learning rate finder_ dan tentukan berapa nilai _learning rate_ yang akan digunakan. Ingat, kita tidak ingin menggunakan LR yang nilainya sama. Hal ini karena layer awal model hanya mengenali sudut, garis, dll. Sementara itu untuk layer terakhir kita ingin untuk lebih diupdate.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Ambil LR sebelum terjadi peningkatan _loss_

# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-5, 1e-4))


# Akurasi model meningkat hingga 93.9%.

# Setelah itu, lihat interpretasi model.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)

