#!/usr/bin/env python
# coding: utf-8

# # Masline 10.07.

# Ovdje importamo fast.ai library i podesimo parametre za grafove (nista bitno :))

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai import *
from fastai.vision import *


# [](http://)Kopiramo folder sa 448 velicinom iz input foldera koji je ReadOnly u working folder. 
# 
# Ako se zelite igrati sa folderima i navigirati, ovo su komande:
# `cd /ime` ulazi u folder `ime`
# `cd ..` ide jedan folder nazad
# `cd /` ide u `root` folder ili prvi folder na disku
# `cd /kaggle/working/` ce otici u kaggle working
# `ls` ce ispisati sve foldere u folderu u kojem jeste.
# 
# ako odemo u input folder, vidimo koje sve velicine dataseta imamo
# 

# In[ ]:


cd /kaggle/input/


# In[ ]:


ls


# In[ ]:


cp -R /kaggle/input/mk0907-448/mk0907-448 /kaggle/working/


# Provjerimo je li zaista kopiran. 

# In[ ]:


cd /kaggle/working/


# In[ ]:


ls


# spremimo u `path` varijablu put do naseg odabranog foldera. tako da kada god zelimo koristiti taj folder opet, samo pozovemo path.
# na path isto mozemo pozivati ls() da vidimo sta ima u tom folderu.

# In[ ]:


path = Path('/kaggle/working/mk0907-448')
path.ls()


# # Dataset

# Idemo inicijalizirati neke parametre sa kojima se mozemo igrati.
# 
# `bs` je batch size. On utvrdjuje koliko uzimamo slika u isto vrijeme u graficku kartu. Ako budemo radili sa slikama vece rezolucije, mozda cemo trebati smanjiti sa 64
# `ds_tfms` su transformacije koje se primjenjuju na slike.
# `size` je velicina slika 

# In[ ]:


bs = 64


# In[ ]:


ds_tfms = get_transforms(flip_vert=True,max_lighting=0.2,max_zoom=1.3,max_rotate=30)


# `data` je varijabla u koju spremamo nas inicijalizirani dataset. Posto radimo transfer learning na modelu koji je vec istreniran na imagenet slikama i poznaje puno razlicitih objekata, moramo primjeniti iste tonove na nase slike (npr, da je zelena boja kod nas ista zelena kao i kod njih)
# 
# `data.show_batch` ce nam pokazati par slika iz naseg novog dataseta

# In[ ]:


data = ImageDataBunch.from_folder(path,ds_tfms=ds_tfms,size=448,bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(12,12))


# In[ ]:


print(data.classes)


# # Learner

# Learner je objekt koji ce vrsiti treniranje za nas. Njemu predajemo `data` koji sadrzi nase slike, model koji je vec istreniran `(models.resnet50, models.resnet34, models.resnet100)`. Broj nakon resneta govori koliko ima slojeva model koji mi koristimo.
# `metrics` ne utjece na treniranje, nego nam ispisuje koliko smo uspjesni. npr ako stavimo `error_rate` dobiti cemo postotak greske, a sada imamo `accuracy` koji kaze postotak uspjeha.

# In[ ]:


learn = cnn_learner(data, models.resnet34,metrics=accuracy)


# Spremimo prazan model, tako da ako ikada zelimo ici ispocetka, samo pokrenemo `learn.load(empty)`

# In[ ]:


learn.save('empty')


# In[ ]:


learn.load('empty')


# `lr_find` pronalazi learning rate funkciju. Zelimo uzeti onaj broj gdje je nagib najveci. mozemo eksperimentirati sa * 10 vecim ili sa * 10 manjim.
# 
# 1e-3 znaci 0.001 
# 
# 10 puta veci je 0.01 ili 1e-2
# 
# 10 puta manji je 0.0001 ili 1e-4
# 
# mozete napisati i sa tockom i sa 1e-5
# 
# 0.005 je onda 5e-3 ili pola izmedju 1e-2 i 1e-3

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# s ovim se mozete igrati i mijenjati. krenete od `empty` i vidite kako vam baca drugaciji accuracy za razlicite lrove, i za razliciti broj ciklusa
# sto je veci lr (a veci je ako je manji broj poslije minusa 1e-3 < 1e-2 :)) to nam brze ide ucenje. sada nam kroz 7 ciklusa dodje na preko 90%. probajte se igrati sa lr-om, da u manje ciklusa dodjete do 95.

# In[ ]:


lr=1e-3
cycle=7


# In[ ]:


learn.fit_one_cycle(cycle,lr)


# Spremiti cemo model, i idemo na fine tuning

# In[ ]:


learn.save('stage1')


# In[ ]:


learn.load('stage1')


# Unfreeze ce nam dopustiti da treniramo i dublje slojeve Neuronske mreze, do sada smo samo trenirali "Glavu". iliti zadnjih par slojeva.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# ovdje predajemo manji learning rate nego u treniranju glave, jer smo vec jako blizu minimuma i trazimo male pomake. `Slice` ce odrediti u kojem ce rasponu lr-a uciti. Prvi broj zelimo uzeti prije nego skoci gore funkcija na grafu, a u ovom slucaju je to `1e-5`. Drugi broj moze biti stari `lr/5` ili stari `lr/10`  kao neki broj koji cesto funkcionira. naravno, mozete se igrati s time. ako zelite samo testirati fine tuning, krenete od `load('stage1')`

# In[ ]:


learn.fit_one_cycle(3,slice(1e-5,lr/5))


# Spremimo model i idemo vidjeti gdje smo fulali.
# 

# In[ ]:


learn.save('stage2')


# # Interpretacija

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(dpi=120)


# In[ ]:


interp.plot_top_losses(4,largest=True) #moze biti i `False` da ide od najtocnijih


# Eto nam slika da vidimo zasto. Tri su rosignole pretamne, a carbonaca je prelomljena :)
# Probajte slobodno sa drugim velicinama, mozda ce bit bolji accuracy sa 896.
