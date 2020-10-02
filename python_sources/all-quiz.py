#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from xlrd import open_workbook

print(14 * " >", "\t n.B.a. \t", "< " * 14, "\n\n\n")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ##### BUTUN EXCEL DOSYALARININ ICINI OKUYALIM #####

# In[ ]:




df = pd.ExcelFile("/kaggle/input/quiz1.xlsx")

quiz1_df_sheets = []                                                 # dosya icindeki sheetlerin isimlerinin tutuldugu liste
quiz1_df_sheet_names = []                                            # data frame haline gelen df_quiz_sheet listesindeki isimlerin tekrar kullanalabilmek icin
total_df_sheetnames = []                                             # quiz1_df_sheets icindeki elemanlarin dataframe hallerinin guncel hali

[quiz1_df_sheets.append(i) for i in df.sheet_names]                  # dosyadaki sheet isimlerini bulur listeye kaydeder
[quiz1_df_sheet_names.append(i) for i in quiz1_df_sheets]            # sheet isimlerini yedekler

for i in range(len(quiz1_df_sheets)):                                                                   # sheetleri data frame haline guncelleyerek degistirir
    quiz1_df_sheets[i] = pd.read_excel("/kaggle/input/quiz1.xlsx", sheet_name=str(quiz1_df_sheets[i]))
    quiz1_df_sheets[i]['Class'] = [quiz1_df_sheet_names[i] for j in range(len(quiz1_df_sheets[i]))]     # sheetler data frame haline gelirken Class adli bir sutunda ekleniyor
    total_df_sheetnames.append(quiz1_df_sheets[i])

quiz1_all = pd.concat([i for i in total_df_sheetnames], sort=False)                                   # tum sheetleri tek dataframe haline getirir
quiz1_all.index = np.arange(1, len(quiz1_all)+1)                                                    # quiz1_all frame sabit index atanmasi yapildi
quiz1_all.columns = ["Name", 'True', 'False', 'Empty', 'Class']                                       # dataframe sutun isimleri degistirildi
quiz1_all["Name"] = list(map(lambda x: x.lower(), quiz1_all["Name"]))                               # butun isimler duzgun olsun diye kucuk hale getirildi
quiz1_all.replace(to_replace='Not Entered', value=0, inplace=True)                                    # yani kolonlari 0 ile 'Not Entered' degistirildi
quiz1_all = quiz1_all.fillna(0)                                                                     # 'NaN' kolonlari 0 ile degistirildi

quiz1_all = quiz1_all[["Class", "Name", 'True', 'False', 'Empty']]          # Class sutunu ensondaydi en basa almis olduk. Burada [[]] yapmammizin sebebi dataframe.core donmesi icin
quiz1_all["Name"] = list(map(lambda x: x.lower(), quiz1_all["Name"]))       # isimlerin hepsini kucuk yapiyoruz

print(quiz1_all)


# ################%% quiz2 SINAV OKUMASI BASLANGIC#####################
# 
# quiz2 dosyasinin icindeki bilgiler excel icine bakildiginda kesinlikle quiz1 den farkli doldurulmus dolayisiyla etap etap dataframe donusturmeliyiz

# In[ ]:


mind_exel = open_workbook("/kaggle/input/py_mind.xlsx")
sense_exel = open_workbook("/kaggle/input/py_sense.xlsx")
science_exel = open_workbook("/kaggle/input/py_science.xlsx")
opinion_exel = open_workbook("/kaggle/input/py_opinion.xlsx")

#the students of classes
mind_students = mind_exel.sheet_names()
sense_students = sense_exel.sheet_names()
science_students = science_exel.sheet_names()
opinion_students = opinion_exel.sheet_names()

mind_len = len(mind_students)-1
sense_len = len(sense_students)-1
science_len = len(science_students)-1
opinion_len = len(opinion_students)-1

mind_sheets = 0
sense_sheets = 0
science_sheets = 0
opinion_sheets = 0

quiz2_all = pd.DataFrame(columns=("Class", "Name", "True", "False", "Empty"), index=None)

while mind_sheets < mind_len:
    mind_sheets += 1
    mind_ex = pd.read_excel("/kaggle/input/py_mind.xlsx", sheet_name=mind_sheets)
    mind_results = mind_ex.tail(3)
    # print(mind_results)
    class_name = "py_mind"
    name = mind_results.columns[0]
    true = mind_results.iloc[0,1]
    # print(true)
    false = mind_results.iloc[1,1]
    empty = mind_results.iloc[2,1]
    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty}
    quiz2_all = quiz2_all.append(student, ignore_index=True)

while sense_sheets < sense_len:
    sense_sheets += 1
    sense_ex = pd.read_excel("/kaggle/input/py_sense.xlsx", sheet_name=sense_sheets)
    sense_results = sense_ex.tail(3)
    class_name = "py_sense"
    name = sense_results.columns[0]
    name = str(name)
    true = sense_results.iloc[0,1]
    false = sense_results.iloc[1,1]
    empty = sense_results.iloc[2,1]
    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty}
    quiz2_all = quiz2_all.append(student, ignore_index=True)

while science_sheets < science_len:
    science_sheets += 1
    science_ex = pd.read_excel("/kaggle/input/py_science.xlsx", sheet_name=science_sheets)
    science_results = science_ex.tail(3)
    class_name = "py_science"
    name = science_results.columns[0]
    name = str(name)
    true = science_results.iloc[0,1]
    false = science_results.iloc[1,1]
    empty = science_results.iloc[2,1]
    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty}
    quiz2_all = quiz2_all.append(student, ignore_index=True)

while opinion_sheets < opinion_len:
    opinion_sheets += 1
    opinion_ex = pd.read_excel("/kaggle/input/py_opinion.xlsx", sheet_name=opinion_sheets)
    opinion_results = opinion_ex.tail(3)
    class_name = "py_opinion"
    name = opinion_results.columns[0]
    name = str(name)
    true = opinion_results.iloc[0,1]
    false = opinion_results.iloc[1,1]
    empty = opinion_results.iloc[2,1]
    student = {'Class': class_name, 'Name': name, 'True': true, 'False': false, "Empty": empty}
    quiz2_all = quiz2_all.append(student, ignore_index=True)


quiz2_all["Name"] = list(map(lambda x: x.lower(), quiz2_all["Name"]))                               # isimlerin hepsini kucuk yapiyoruz

print(quiz2_all)


# ################%% quiz3 SINAV OKUMASI BASLANGIC#####################

# In[ ]:


df2 = pd.read_excel("/kaggle/input/sinav-2.xlsx")

# print(df2)

quiz3_all = df2.iloc[1:, 1:6]                                                   # sutunlari daha sade hale getirmek icin kesme yaptik
quiz3_all.loc[quiz3_all['class'] == 'sense', ['class']] = "py_sense"                # sutun isimlerini degistirdik
quiz3_all.loc[quiz3_all['class'] == 'science', ['class']] = "py_science"
quiz3_all.loc[quiz3_all['class'] == 'mind', ['class']] = "py_mind"
quiz3_all.loc[quiz3_all['class'] == 'opinion', ['class']] = "py_opinion"

quiz3_all.rename(columns={"class": "Class", "name": "Name", "true": "True", "false": "False", "empty": "Empty"}, inplace=True)          # sutun isimlerini degistik

quiz3_all["Name"] = list(map(lambda x: x.lower(), quiz3_all["Name"]))                               # isimlerin hepsini kucuk yapiyoruz
print(quiz3_all)



# ########### quiz4 OKUMA BASLANGIC ##########################

# In[ ]:


quiz4_all = pd.read_excel("/kaggle/input/uygulamali.xlsx")

# quiz4_all = quiz4_all.iloc[:, :]
columns = ['1.', '2.',  '3.', '4.',  '5.', '6.', '7.', ]                       # listelerde sonlara virgul koyulmasi tavsiye edilir
quiz4_all.drop(columns, inplace=True, axis=1)                                  # sutun isimleri olan 1 2 3, ... sildik
quiz4_all.insert(loc=1, column='False', value=0)                               # diger sinavlardaki sutunlarda true false empty oldugu icin ve burda kullanilmadigindan degeri sifir olan bu sutunlari ekledik
quiz4_all.insert(loc=2, column='Empty', value=0)
quiz4_all.rename(columns={"isim": "Name", "sinif": "Class", "total": "True"}, inplace=True)             # sutun isimlerini degistirdik
quiz4_all = quiz4_all[["Class", "Name", 'True', 'False', 'Empty']]                                      # Class sutunu ensondaydi en basa almis olduk. Burada [[]] yapmammizin sebebi dataframe.core donmesi icin
quiz4_all["Name"] = list(map(lambda x: x.lower(), quiz4_all["Name"]))                                   # isimlerin hepsini kucuk yapiyoruz


print(quiz4_all)


# #################### BUTUN quizler birlestirilip tek dosya haline DATAFRAME donusturuldu ##########################

# In[ ]:


all_quiz = pd.DataFrame(columns=("Class", "Name", 'True', 'False', 'Empty'), index=None)
all_quiz = all_quiz.append(quiz1_all, ignore_index=True)
all_quiz = all_quiz.append(quiz2_all, ignore_index=True)
all_quiz = all_quiz.append(quiz3_all, ignore_index=True)
all_quiz = all_quiz.append(quiz4_all, ignore_index=True)

print(all_quiz)


# ##################################### Siniflarina gore True False Empty ###################################

# In[ ]:


class_sense = all_quiz[all_quiz["Class"] == "py_sense"]                                 # siniflara ayrildi
class_science = all_quiz[all_quiz["Class"] == "py_science"]
class_opinion = all_quiz[all_quiz["Class"] == "py_opinion"]
class_mind = all_quiz[all_quiz["Class"] == "py_mind"]

true_mean_sense = class_sense["True"].mean()                                                 # siniflara gore True ortalamalari alindi
true_mean_science = class_science["True"].mean()
true_mean_opinion = class_opinion["True"].mean()
true_mean_mind = class_mind["True"].mean()

false_mean_sense = class_sense["False"].mean()                                                 # siniflara gore False ortalamalari alindi
false_mean_science = class_science["False"].mean()
false_mean_opinion = class_opinion["False"].mean()
false_mean_mind = class_mind["False"].mean()

empty_mean_sense = class_sense["Empty"].mean()                                                 # siniflara gore Empty ortalamalari alindi
empty_mean_science = class_science["Empty"].mean()
empty_mean_opinion = class_opinion["Empty"].mean()
empty_mean_mind = class_mind["Empty"].mean()


sorting_classes = sorted([true_mean_sense, true_mean_mind, true_mean_science, true_mean_opinion])           # siniflarin dogru siralamalari
# print(sorting_classes)

# best_class_student_sense = class_sense.sort_values(by="True", ascending=False).head(1)          # en iyi ogrenci burada anlam ifade etmez cunku ad soyad tam girilmemis



# ################################### Standart sapma ########################3

# In[ ]:


print("\n", 14 * " >", "\t Standart deviation of Classes \t", "< " * 14, "\n")


std_sense = np.std([true_mean_sense, false_mean_sense, empty_mean_sense])
std_science = np.std([true_mean_science, false_mean_science, empty_mean_science])
std_mind = np.std([true_mean_mind, false_mean_mind, empty_mean_mind])
std_opinion = np.std([true_mean_opinion, false_mean_opinion, empty_mean_opinion])
print("py_sense std deviation: ", std_sense, "\npy_science std deviation: ", std_science,
      "\npy_mind std deviation: ", std_mind, "\npy_opinon std deviation: ", std_opinion)


# ############################# MATPLOTLIB GRAFIK & GORSELLESTIRME ##################################

# In[ ]:


y = np.array([true_mean_opinion, true_mean_science, true_mean_mind, true_mean_sense])
x = np.array([1, 2, 3, 4])


# In[ ]:


############## bar plot #################################33

arr = np.arange(5)
plt.bar(x, y)
plt.title("bar plot for True numbers according to Classes")
plt.xlabel("x stick")
plt.ylabel("mean stick")
plt.xticks(arr, (' ', "py_opinion", "py_science", "py_mind", "py_sense"))              # siniflarin ortalamalari etiket verme x eksenine gore
plt.show()


# In[ ]:


######################## siniflarin standart sapmalarina gore bar plot grafigi  ############################3

x = np.array([1, 2, 3, 4])
arr = np.arange(5)
plt.bar(x, y)
plt.title("Standart deviation bar plot according to Classes")
plt.xlabel("x stick")
plt.ylabel("std deviation stick")
plt.xticks(arr, (" ", "py_mind", "py_sense", "py_science", "py_opinion"))
plt.show()


# In[ ]:


################ Siniflarin True sayilarina gore PLOT CIZIMI ###############################3

plt.plot(all_quiz[all_quiz["Class"] == "py_mind"]["True"], color="red", label="py_mind")
plt.plot(all_quiz[all_quiz["Class"] == "py_sense"]["True"], color="blue", label="py_sense")
plt.plot(all_quiz[all_quiz["Class"] == "py_science"]["True"], color="black", label="py_science")
plt.plot(all_quiz[all_quiz["Class"] == "py_opinion"]["True"], color="green", label="py_opinion")
plt.legend()                                                # legend x ve y cubuklarini olusturuyor
plt.xlabel("Classes")
plt.ylabel("According to True")
plt.show()


# In[ ]:


################ Siniflarin True sayilarina gore SCATTER CIZIMI ###############################3

plt.scatter(all_quiz[all_quiz["Class"] == "py_mind"]["True"].index, all_quiz[all_quiz["Class"] == "py_mind"]["True"], color="red", label="py_mind")
plt.scatter(all_quiz[all_quiz["Class"] == "py_sense"]["True"].index, all_quiz[all_quiz["Class"] == "py_sense"]["True"], color="blue", label="py_sense")
plt.scatter(all_quiz[all_quiz["Class"] == "py_science"]["True"].index, all_quiz[all_quiz["Class"] == "py_science"]["True"], color="black", label="py_science")
plt.scatter(all_quiz[all_quiz["Class"] == "py_opinion"]["True"].index, all_quiz[all_quiz["Class"] == "py_opinion"]["True"], color="green", label="py_opinion")
plt.legend()
plt.title("Scatter plot")
plt.xlabel("Classes")
plt.ylabel("According to True")
plt.show()

