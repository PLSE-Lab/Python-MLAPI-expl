#!/usr/bin/env python
# coding: utf-8

# # Pelatihan Data Science Workshop PPI Hsinchu  
# 
# Kernel Asli : https://www.kaggle.com/andresionek/what-makes-a-kaggler-valuable/notebook?utm_medium=social&utm_source=twitter.com&utm_campaign=Weekly-Kernel-Awards
# 
# Selamat Datang di python jupiter notebook! Kaggle menyediakan komputasi online gratis dimana setiap sesi memiliki batasan:
# Run time : 6 Hours
# Memory : 16 GB
# Environtment : https://github.com/Kaggle/docker-python/blob/master/Dockerfile
# 
# Dalam sesi komputasi ini kita tidak perlu melakukan instalasi apapun lagi karena hampir semua libray penting sudah di sediakan dalam environtmentnya. Untuk tahu lebih jelas apa yang sudah terinstall bisa akses link environtment diatas. Kita juga diberi batasan hanya diperbolehkan menjalankan komputasi sebanyak maksimal 25 script saja dalam waktu yang bersamaan. 
# Ini bisa dimanfaatkan untuk mencoba berbagai macam ide nantinya kalau kita ingin mencoba ide mana yang berhasil dan tidak dalam iterasi model matematis. 
# 
# ## Survey Komunitas Kaggle 2018  
# 
# Seperti yang sudah dijelaskan sebelumnya kita tidak akan bermain langsung dengan data finhacks dikarenakan saya tidak diperbolehkan untuk menyebarluaskan data dan model dalam bentuk apapun, oleh karena itu kita akan mencoba menerapkan dan berlatih teknik teknik dasar yang saya ketahui dengan menggunakan dataset hasil survey komunitas kaggle 2018 ini.
# 
# Survey ini di isi oleh 23,859 responden yang pengisi survey diberi pertanyaan seputar kegiatan dan posisi mereka sehari hari yang memiliki keterkaitan dengan datascience. Seperti framework apa yang mereka gunakan dan lain sebagainya. 
# 
# Sebelum memulai ada baiknya untuk mengamati sejenak semua elemen yang ada di komputer ini.
# 
# ## Memuat Data  
# 
# Langkah pertama dalam setiap analisis data adalah memuat datanya sendiri, Disini kita menggunakan "pandas" sebagai data manipulator. Jalankan cell berikut untuk load data ke dalam environtment.
# 

# In[ ]:


import numpy as np 
import pandas as pd
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)

# Loading the multiple choices dataset, we will not look to the free form data on this study
mc = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory=False)

# Separating questions from answers
# This Series stores all questions
mcQ = mc.iloc[0,:]
# This DataFrame stores all answers
mcA = mc.iloc[1:,:]


# Untuk mengetahui summary dari data yang kita miliki jalankan :

# In[ ]:


print(mcQ.shape)
mcQ.describe()


# In[ ]:


print("Jumlah Kolom = {}".format(mcA.shape[1]))
print("Jumlah Baris = {}".format(mcA.shape[0]))
mcA.describe()


# Untuk melihat seperti apa data sesungguhnya silahkan run:

# In[ ]:


mcA.head(2)


# In[ ]:


# Set ipython's max row display
pd.set_option('display.max_row', 1000)
# Set iPython's max column width to 50
pd.set_option('display.max_columns', 50)
pd.set_option('max_colwidth',100)
mcQ


# ## Simplfikasi Data 
# 
# Untuk mempermudah proses analisa sebagian besar data kita drop saja agar pelatihan bisa lebih simple dan dapat dengan mudah dipahami.

# In[ ]:


# Creating a table with personal data
personal_data = mcA.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,22,84,86,108,124,126,127,128,129]].copy()

# renaming columns
cols = ['survey_duration', 'gender', 'gender_text', 'age', 'country', 'education_level', 'undergrad_major', 'role', 'role_text',
        'employer_industry', 'employer_industry_text', 'years_experience', 'yearly_compensation','primary_analize_tool', 'most_used_prog_lang',
        'most_reccom_prog_lang','most_used_ml_lib','most_used_vis_lib','coding_time','coding_time_years','ml_exp_years','are_you_data_scientist']
personal_data.columns = cols

# Drop text based features
personal_data.drop(['gender_text', 'role_text', 'employer_industry_text'], axis=1, inplace=True)


personal_data.head(5)


# In[ ]:


personal_data.shape


# Ternyata ada data NaN, coba kita cek Nan untuk semua tipe variable, run:

# In[ ]:


personal_data.isnull().any()


# In[ ]:


personal_data['survey_duration'].unique()


# In[ ]:


personal_data['gender'].unique()


# In[ ]:


personal_data['age'].unique()


# In[ ]:


personal_data['country'].unique()


# In[ ]:


personal_data['education_level'].unique()


# In[ ]:


personal_data['undergrad_major'].unique()


# In[ ]:


personal_data['role'].unique()


# In[ ]:



personal_data['employer_industry'].unique()


# In[ ]:



personal_data['years_experience'].unique()


# In[ ]:



personal_data['yearly_compensation'].unique()


# In[ ]:



personal_data['primary_analize_tool'].unique()


# In[ ]:



personal_data['most_used_prog_lang'].unique()


# In[ ]:



personal_data['most_reccom_prog_lang'].unique()


# In[ ]:



personal_data['most_used_ml_lib'].unique()


# In[ ]:



personal_data['most_used_vis_lib'].unique()


# In[ ]:



personal_data['coding_time'].unique()


# In[ ]:



personal_data['coding_time_years'].unique()


# In[ ]:



personal_data['ml_exp_years'].unique()


# In[ ]:



personal_data['are_you_data_scientist'].unique()


# ## Objective
# 
# Kita hendak membangun model dimana dengan memasukkan latar belakang ilmu data science seseorang kita bisa menebak berapa persen kemungkinannya menerima gaji sebesar 100.000 keatas.
# 
# ## Data Cleaning
# 
# ### Missing value
# 
# Sudah kita amati bersama banyak kolom / variabel yang mengandung data NaN, langkah paling simple adalah mendelete data yang mengandung NaN. Tapi tidak selalu menghapus data dengan NaN itu merupakan langkah yang bagus. Di sini coba kita hapus data yang mengandung NaN tapi hanya untuk kolom Gaji. Karena kolom ini yang akan kita gunakan sebagai target latihan.
# 

# In[ ]:


# dropping all NaN and I do not wish to disclose my approximate yearly compensation, because we are only interested in respondents that revealed their earnings
personal_data = personal_data[~personal_data['yearly_compensation'].isnull()].copy()
not_disclosed = personal_data[personal_data['yearly_compensation'] == 'I do not wish to disclose my approximate yearly compensation'].index
personal_data = personal_data.drop(list(not_disclosed), axis=0)
print("Done!")


# In[ ]:


compensation = personal_data.yearly_compensation.str.replace(',', '').str.replace('500000\+', '500-500000').str.split('-')
personal_data['yearly_compensation_numerical'] = compensation.apply(lambda x: (int(x[0]) * 1000 + int(x[1]))/ 2) / 1000 # it is calculated in thousand dollars
#personal_data = personal_data.drop(['yearly_compensation'], axis=1)
print('Dataset Shape: ', personal_data.shape)
personal_data.head(3)


# ---
# # Exploratory Data Analysis 
# 
# Sekarang mari kita coba plot informasi informasi yang ada didalam dataset ini. Kernel asli sudah menyediakan banyak sekali bantuan untuk mempermudah kita dalam membuat grafik yang cantik. Fungsi fungsi yang disediakan menggunakan plotly sebagai library untuk plotting.
# 
# ### Finding the Top 20% most well paid

# In[ ]:


# Finding the compensation that separates the Top 20% most welll paid from the Bottom 80%
top20flag = personal_data.yearly_compensation_numerical.quantile(0.8)
top20flag


# In[ ]:


# Creating a flag to identify who belongs to the Top 20%
personal_data['top20'] = personal_data.yearly_compensation_numerical > top20flag

# creating data for future mapping of values
top20 = personal_data.groupby('yearly_compensation', as_index=False)['top20'].min()
print("Done!")


# In[ ]:


# Some helper functions to make our plots cleaner with Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)


def gen_xaxis(title):
    """
    Creates the X Axis layout and title
    """
    xaxis = dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            )
    return xaxis


def gen_yaxis(title):
    """
    Creates the Y Axis layout and title
    """
    yaxis=dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            )
    return yaxis


def gen_layout(charttitle, xtitle, ytitle, lmarg, h, annotations=None):  
    """
    Creates whole layout, with both axis, annotations, size and margin
    """
    return go.Layout(title=charttitle, 
                     height=h, 
                     width=800,
                     showlegend=False,
                     xaxis=gen_xaxis(xtitle), 
                     yaxis=gen_yaxis(ytitle),
                     annotations = annotations,
                     margin=dict(l=lmarg),
                    )


def gen_bars(data, color, orient):
    """
    Generates the bars for plotting, with their color and orient
    """
    bars = []
    for label, label_df in data.groupby(color):
        if orient == 'h':
            label_df = label_df.sort_values(by='x', ascending=True)
        if label == 'a':
            label = 'lightgray'
        bars.append(go.Bar(x=label_df.x,
                           y=label_df.y,
                           name=label,
                           marker={'color': label},
                           orientation = orient
                          )
                   )
    return bars


def gen_annotations(annot):
    """
    Generates annotations to insert in the chart
    """
    if annot is None:
        return []
    
    annotations = []
    # Adding labels
    for d in annot:
        annotations.append(dict(xref='paper', x=d['x'], y=d['y'],
                           xanchor='left', yanchor='bottom',
                           text= d['text'],
                           font=dict(size=13,
                           color=d['color']),
                           showarrow=False))
    return annotations


def generate_barplot(text, annot_dict, orient='v', lmarg=120, h=400):
    """
    Generate the barplot with all data, using previous helper functions
    """
    layout = gen_layout(text[0], text[1], text[2], lmarg, h, gen_annotations(annot_dict))
    fig = go.Figure(data=gen_bars(barplot, 'color', orient=orient), layout=layout)
    return iplot(fig)

print("Done!")


# In[ ]:


from pandas.api.types import CategoricalDtype
# transforming compensation into category type and ordering the values
categ = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000',
         '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
         '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000',
         '300-400,000', '400-500,000', '500,000+']
cat_type = CategoricalDtype(categories=categ, ordered=True)
personal_data.yearly_compensation = personal_data.yearly_compensation.astype(cat_type)
# Kode diatas untuk memastikan bahwa urutan penampilannya sesuai dengan yang kita inginkan walaupun kita sudah menset sebagai kategori diatas

# Counting the quantity of respondents per compensation
barplot = personal_data.yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20% label
barplot = barplot.merge(top20, on='yearly_compensation')
barplot.columns = ['x', 'y', 'top20']

# apply color for top 20% and bottom 80%
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray') 

# Create title and annotations
title_text = ['<b>How Much Does Kagglers Get Paid?</b>', 'Yearly Compensation (K USD)', 'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 2200, 'text': '80% of respondents earn up to USD 90k','color': 'gray'},
              {'x': 0.51, 'y': 1100, 'text': '20% of respondents earn more than USD 90k','color': 'mediumaquamarine'}]

# call function for plotting
generate_barplot(title_text, annotations)

print("Done!")


# Bagaimana dengan pelajar? Dimana posisi mereka?

# In[ ]:


# creating masks to identify students and not students
is_student_mask = (personal_data['role'] == 'Student') | (personal_data['employer_industry'] == 'I am a student')
not_student_mask = (personal_data['role'] != 'Student') & (personal_data['employer_industry'] != 'I am a student')

# Counting the quantity of respondents per compensation (where is student)
barplot = personal_data[is_student_mask].yearly_compensation.value_counts(sort=False).to_frame().reset_index()
barplot.columns = ['yearly_compensation', 'qty']

# mapping back to get top 20% label
barplot = barplot.merge(top20, on='yearly_compensation')
barplot.columns = ['x', 'y', 'top20']

# apply color for top 20% and bottom 80%
barplot['color'] = barplot.top20.apply(lambda x: 'mediumaquamarine' if x else 'lightgray') 

# title and annotations
title_text = ['<b>Do Students Get Paid at All?</b><br><i>only students</i>', 'Yearly Compensation (K USD)', 'Quantity of Respondents']
annotations = [{'x': 0.06, 'y': 1650, 'text': '75% of students earn up to USD 10k','color': 'crimson'}]

# ploting
generate_barplot(title_text, annotations)


# ### Should you get formal education?
# Seberapa penting sih sekolah tinggi tinggi dalam dunia data science in terms of seberapa tinggi gaji pada umumnya

# In[ ]:


# Calculates compensation per education level
barplot = personal_data[not_student_mask].groupby(['education_level'], as_index=False)['yearly_compensation_numerical'].mean()
barplot['no_college'] = (barplot.education_level == 'No formal education past high school') |                         (barplot.education_level == 'Doctoral degree')

# creates a line break for better visualisation
barplot.education_level = barplot.education_level.str.replace('study without', 'study <br> without')

barplot.columns = ['y', 'x', 'no_college']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.no_college.apply(lambda x: 'coral' if x else 'a')

# Add title and annotations
title_text = ['<b>Impact of Formal Education on Compenstaion</b><br><i>without students</i>', 'Average Yearly Compensation (K USD)', 'Level of Education']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300)


# ### Which industry should you target?
# Industri apa saja dalam penerapan data science yang berani membayar mahal?

# In[ ]:


# Calculates compensation per industry
barplot = personal_data[not_student_mask].groupby(['employer_industry'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 industries to add color
barplot['best_industries'] = (barplot.employer_industry == 'Medical/Pharmaceutical') |                              (barplot.employer_industry == 'Insurance/Risk Assessment') |                              (barplot.employer_industry == 'Military/Security/Defense') |                              (barplot.employer_industry == 'Hospitality/Entertainment/Sports') |                              (barplot.employer_industry == 'Accounting/Finance')

barplot.columns = ['y', 'x', 'best_industries']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.best_industries.apply(lambda x: 'darkgoldenrod' if x else 'a')

title_text = ['<b>Average Compensation per Industry | Top 5 in Color</b><br><i>without students</i>', 'Average Yearly Compensation (K USD)', 'Industry']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)


# ### Should You Aim at the C-level?
# CFO, CEO, dan C C lainnya seberapa besar gaji mereka di compare dengan pekerja bawahan seperti saya?

# In[ ]:


# Calculates compensation per role
barplot = personal_data[not_student_mask].groupby(['role'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 roles to add color
barplot['role_highlight'] = (barplot.role == 'Data Scientist') |                         (barplot.role == 'Product/Project Manager') |                         (barplot.role == 'Consultant') |                         (barplot.role == 'Data Journalist') |                         (barplot.role == 'Manager') |                         (barplot.role == 'Principal Investigator') |                         (barplot.role == 'Chief Officer')

barplot.columns = ['y', 'x', 'role_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.role_highlight.apply(lambda x: 'mediumvioletred' if x else 'lightgray')

title_text = ['<b>Average Compensation per Role | Top 7 in Color</b><br><i>without students</i>', 'Average Yearly Compensation (USD)', 'Job Title']
annotations = [{'x': 0.6, 'y': 11.5, 'text': 'The first step into the ladder<br>of better compensation is<br>becoming a Data Scientist','color': 'mediumvioletred'}]

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)


# In[ ]:


# Calculates compensation per role
barplot = personal_data[not_student_mask].groupby(['most_used_prog_lang'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 5 roles to add color

barplot['role_highlight'] = (barplot.most_used_prog_lang == 'Scala') 

barplot.columns = ['y', 'x','role_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.role_highlight.apply(lambda x: 'mediumvioletred' if x else 'lightgray')

title_text = ['<b>Most Used Programming Language</b><br><i>without students</i>', 'Average Yearly Compensation (K USD)', 'Language']
annotations = [{'x': 0.8, 'y': 11.5, 'text': 'Scalla, Go, <br> Aduh saya ga tahu','color': 'mediumvioletred'}]

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=600)


# ### Which countries pay more?
# Negara mana yang paling bersahabat dengan data scientist? (Gajinya gede maksudnya)

# In[ ]:


# Replacing long country names
personal_data.country = personal_data.country.str.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
personal_data.country = personal_data.country.str.replace('United States of America', 'United States')
personal_data.country = personal_data.country.str.replace('I do not wish to disclose my location', 'Not Disclosed')
personal_data.country = personal_data.country.str.replace('Iran, Islamic Republic of...', 'Iran')
personal_data.country = personal_data.country.str.replace('Hong Kong \(S.A.R.\)', 'Hong Kong')
personal_data.country = personal_data.country.str.replace('Viet Nam', 'Vietnam')
personal_data.country = personal_data.country.str.replace('Republic of Korea', 'South Korea')

# Calculates compensation per country
barplot = personal_data[not_student_mask].groupby(['country'], as_index=False)['yearly_compensation_numerical'].mean()

# Flags the top 10 countries to add color
barplot['country_highlight'] = (barplot.country == 'United States') |                                (barplot.country == 'Switzerland') |                                (barplot.country == 'Australia') |                                (barplot.country == 'Israel') |                                (barplot.country == 'Denmark') |                                (barplot.country == 'Canada') |                                (barplot.country == 'Hong Kong') |                                (barplot.country == 'Norway') |                                (barplot.country == 'Ireland') |                                (barplot.country == 'United Kingdom')

barplot.columns = ['y', 'x', 'country_highlight']
barplot = barplot.sort_values(by='x', ascending=True)
barplot['color'] = barplot.country_highlight.apply(lambda x: 'mediumseagreen' if x else 'lightgray')

title_text = ['<b>Average Compensation per Country - Top 10 in Color</b><br><i>without students</i>', 'Average Yearly Compensation (USD)', 'Country']
annotations = []

generate_barplot(title_text, annotations, orient='h', lmarg=300, h=1200)


# # Model Building
# 
# Sekarang mari kita coba membuat model matematis untuk memprediksi kemungkinan seseorang termasuk top 20% dari segi gaji.
# Kita review lagi data yang kita punya:

# In[ ]:


personal_data.head(5)


# Kita hapus dulu variable yang jelas jelas merupakan target. Tidak valid model untuk memprediksi gaji tapi meminta inputan gaji kan?

# In[ ]:


model_data = personal_data.drop(['yearly_compensation','yearly_compensation_numerical'], axis=1)
model_data.head(5)


# In[ ]:


model_data.shape


# ## One Hot Encoding
# 
# Nah dari hasil eksplorasi tadi jelas terlihat kalau hampir semua data merupakan data kategori. Kebanyakan model tidak dapat memproses variabel dalam bentuk kategori. Salah satu bentuk feature engineering yang paling umum adalah mengkonversinya kedalam bentuk biner 1 dan 0

# In[ ]:


# All OHE
categorical_features= [
    'gender', 
    'age',
    'country', 
    'education_level', 
    'undergrad_major', 
    'role', 
    'employer_industry', 
    'years_experience', 
    'primary_analize_tool', 
    'most_used_prog_lang', 
    'most_reccom_prog_lang', 
    'most_used_ml_lib', 
    'most_used_vis_lib', 
    'coding_time', 
    'coding_time_years', 
    'ml_exp_years', 
    'are_you_data_scientist',
]
for whichColumn in categorical_features:
    dummy = pd.get_dummies(model_data[whichColumn].astype('category'))
    columns = dummy.columns.astype(str).tolist()
    columns = [whichColumn  + '_' + w for w in columns]
    dummy.columns = columns
    model_data = pd.concat((model_data, dummy), axis=1)
    model_data = model_data.drop([whichColumn], axis=1)
    
model_data.shape


# Mari kita perhatikan seksama bentuk data kita setelah one hot encoding

# In[ ]:


model_data.head(5)


# Saatnya kita pecah data menjadi 3. Train Data, Validation Data dan Test Data

# In[ ]:


from sklearn.model_selection import train_test_split
model_data = model_data.reset_index(drop=True)
model_data.survey_duration = model_data.survey_duration.astype(int)
label  = np.array(model_data['top20'].values.astype(int))
model_data = model_data.drop(['top20'], axis=1)

#Main and Test Split
train_index, test_index = train_test_split(model_data.index.values, shuffle=True, test_size=0.3)
main_data = model_data.iloc[train_index].reset_index(drop=True)
main_label = label[train_index]
test_data  = model_data.iloc[test_index].reset_index(drop=True)
test_label = label[test_index]

#Train and Validation Split
train_index, valid_index = train_test_split(main_data.index.values, shuffle=True, test_size=0.3)
train_data = main_data.iloc[train_index].reset_index(drop=True)
train_label = main_label[train_index]
valid_data  = main_data.iloc[valid_index].reset_index(drop=True)
valid_label = main_label[valid_index]

print("Train Data Shape {}".format(train_data.shape))
print("Valid Data Shape {}".format(valid_data.shape))
print("Test Data Shape {}".format(test_data.shape))


# ## Building a model
# 
# ada banyak macam tipe model. Kita akan mencoba yang simple vs kompleks. 
# Logistic Regresion Vs Gradient Boosted Macine
# 

# In[ ]:


#Simple Model by Logistic Regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
np.random.seed(1)
rn.seed(1)

clf = LogisticRegression(random_state=0).fit(train_data, train_label)

print("Train Data Accuracy {}".format(clf.score(train_data, train_label)))
print("Valid Data Accuracy {}".format(clf.score(valid_data, valid_label)))
print("Test Data Accuracy {}".format(clf.score(test_data, test_label)))


# In[ ]:


#Complex Model by Xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score
np.random.seed(1)
rn.seed(1)

params = {
    #'gamma':2,
    'learning_rate': 0.1, 
    #'eval_metric':'auc',
    'nthread':4, 
    #'max_depth': 18, 
    'booster': 'gbtree', 
    'tree_method':'exact', 
    'objective':'binary:logistic', 
    'seed': 0, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'colsample_bylevel':1,
    'alpha':0, 
    'lambda':0,
    'silent':1,
    'disable_default_eval_metric':1
}

def evalerror(preds, dtrain):
    labels = dtrain.get_label()        
    predsTres = [1 if a >= 0.5 else 0 for a in preds]
    return 'Accuracy', accuracy_score(predsTres,  labels)

d_train = xgb.DMatrix(train_data.values, label=train_label)
d_valid = xgb.DMatrix(valid_data.values, label=valid_label)
d_test = xgb.DMatrix(test_data.values, label=test_label)
watchlist = [(d_train,'Train'), (d_valid,'Valid')]

XGB = xgb.train(params, d_train , 100, watchlist, verbose_eval=10, feval=evalerror)        
resultTrain  = [1 if a >= 0.5 else 0 for a in XGB.predict(d_train)]
resultValid  = [1 if a >= 0.5 else 0 for a in XGB.predict(d_valid)]
resultTest   = [1 if a >= 0.5 else 0 for a in XGB.predict(d_test)]


# In[ ]:


print("Train Data Accuracy {}".format(accuracy_score(resultTrain,  train_label)))
print("Valid Data Accuracy {}".format(accuracy_score(resultValid, valid_label)))
print("Test Data Accuracy {}".format(accuracy_score(resultTest, test_label)))


# Train set error mengecil tapi kenapa valid set error membesar? Ada apa?
