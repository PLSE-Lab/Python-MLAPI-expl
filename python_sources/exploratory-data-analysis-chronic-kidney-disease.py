#!/usr/bin/env python
# coding: utf-8

# ![](https://i0.wp.com/cdn-prod.medicalnewstoday.com/content/images/articles/172/172179/a-cross-section-graphic-of-the-kidneys.jpg?w=1155&h=978)
# 
# # Introduction 
# Our kidney is involved in multiple key functions. 
# * They help to maintain overall fluid balance 
# * They help to regulate and filter minerals from blood
# * They help to filter wastes generated from medications, food and toxic substances
# * They then help in creating hormones that help produce red blood cells, promote bone health, and regulate blood pressure
# 
# ### So what happens if our kidney is damaged?
# We have two kidneys. If one doesn't function well, the burdens get carried over to the second kidney. If the patient fails to take drastic measures to improve his/her condition, both kidneys will fail, leading to acute renal failure. This can be fatal without artificial filtering (dialysis) or a kidney transplant. Of course, this occurs at the advance stage of chronic kidney disease and symptoms will only show up at a severe stage. 
# 
# # Purpose of this project
# - We will like to find out the correlation between these health attributes and chronic kidney disease.
#     * This way we can allow early detections that facilitate medical interventions
# - Identify key precursors to chronic kidney disease that can be used for machine learning 

# ## Importing packages 
# - Here we will import required packages, namely Numpy, Pandas, Matplotlib, Seaborn and Scipy.stats

# In[ ]:


# Importing packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats # Needed to compute statistics for categorical data


# ## Importing dataset into notebook and previewing it

# In[ ]:


kidney_data = pd.read_csv('../input/ckdisease/kidney_disease.csv')
kidney_data.head()


# ## Check for any unknown, NaN or NULL values

# In[ ]:


kidney_data.isnull().sum()


# ## Dealing with NaN values 
# - There are numerous way we can deal with this. Firstly, we can choose to remove all the rows that are associated with the NaN values. Secondly, we can replace the NaN values with the mean or median value of the column. According to [Matt Brems](https://github.com/matthewbrems/ODSC-missing-data-may-18/blob/master/Analysis%20with%20Missing%20Data.pdf), I gathered the following:
#     * If we choose to remove the rows that comprise the NaN value, we are more likely to obtain well-behaved results, usually software-default. However, we do lose some precision in our collected data. 
#     * If we choose to drop no observation and replace them with what is available in the data, we will not encounter well-behaved results (invalid covariance matrices) but we did utilise all the data as intended. 
# 
# - Well, I have decided to remove the rows associated with the NaN values since I am not losing that much data (below my required threshold).

# In[ ]:


kidney_data = kidney_data.dropna(axis=0)
kidney_data.isnull().sum()


# ## Data cleaning
# 
# Looks like we have cleared all the NaN values. Moving on, we will need to clean the data such that we replace all positive data like 'normal', 'positive', 'True', 'yes' with 1 and all negative data('abnormal', negative', 'false', 'no') with 0. 

# In[ ]:


kidney_data['rbc'] = kidney_data.rbc.replace(['normal','abnormal'], ['1', '0'])
kidney_data['pc'] = kidney_data.pc.replace(['normal','abnormal'], ['1', '0'])
kidney_data['pcc'] = kidney_data.pcc.replace(['present','notpresent'], ['1', '0'])
kidney_data['ba'] = kidney_data.ba.replace(['present','notpresent'], ['1', '0'])
kidney_data['htn'] = kidney_data.htn.replace(['yes','no'], ['1', '0'])
kidney_data['dm'] = kidney_data.dm.replace(['yes','no'], ['1', '0'])
kidney_data['cad'] = kidney_data.cad.replace(['yes','no'], ['1', '0'])
kidney_data['appet'] = kidney_data.appet.replace(['good','poor'], ['1', '0'])
kidney_data['pe'] = kidney_data.pe.replace(['yes','no'], ['1', '0'])
kidney_data['ane'] = kidney_data.ane.replace(['yes','no'], ['1', '0'])
kidney_data['classification'] = kidney_data.classification.replace(['ckd','ckd\t','notckd'], ['positive', 'positive','negative'])
kidney_data.head()


# ### Here, we will use the PairPlot tool from Seaborn to see the distribution and relationships among variables.

# In[ ]:


g = sns.pairplot(kidney_data, vars =['age', 'bp','bgr', 'bu', 'sc','sod', 'pot', 'hemo'],hue = 'classification')
g.map_diag(sns.distplot)
g.add_legend()
g.fig.suptitle('FacetGrid plot', fontsize = 20)
g.fig.subplots_adjust(top= 0.9);


# ## What do we see here?
# - We do observe distinct classification of positive and negative result in each targeted health attribute. 
# - Before I continue, the first thing I usually look at is the number of test subjects for each classification, the age and gender(not provided) in order to understand the test subjects involved in this study.  
# - We have fairly similar mean age among these 2 test subjects but we do observe significantly more test subjects tested negative for chronic kidney disease. The data cleaning done to remove NaN values seem to have cleared out a huge amount of test subjects with chronic kidney disease. 
# - At this point I was thinking of replacing my NaN values with the median value of the column but then I realised that the red blood cell(rbc) atribute is a nominal data. 

# In[ ]:


kidney_data[kidney_data['classification'] == 'positive'].describe()


# In[ ]:


kidney_data[kidney_data['classification'] == 'negative'].describe()


# In[ ]:


kidney_data1 = pd.read_csv('../input/ckdisease/kidney_disease.csv')
kidney_data1[kidney_data1['rbc'].isnull()].groupby(['classification']).size().reset_index(name = 'count')


# ## So what can we do?
# - The author of this dataset encourages everyone to omit rows with NaN data. I have decided to utilise the interpolation method from Pandas library to retain the NaN values. 

# In[ ]:


kidney_data = pd.read_csv('../input/ckdisease/kidney_disease.csv')


# In[ ]:


kidney_data = kidney_data.interpolate(method='pad')
kidney_data.rbc = kidney_data.rbc.interpolate(method='pad')
kidney_data.pc = kidney_data.pc.interpolate(method='pad')
kidney_data['rbc'] = kidney_data.rbc.replace(['normal','abnormal'], [1,0])
kidney_data['pc'] = kidney_data.pc.replace(['normal','abnormal'], [1,0])
kidney_data['pcc'] = kidney_data.pcc.replace(['present','notpresent'], [1,0])
kidney_data['ba'] = kidney_data.ba.replace(['present','notpresent'], [1,0])
kidney_data['htn'] = kidney_data.htn.replace(['yes','no'], [1,0])
kidney_data['dm'] = kidney_data.dm.replace(['yes','no'], [1,0])
kidney_data['cad'] = kidney_data.cad.replace(['yes','no'], [1,0])
kidney_data['appet'] = kidney_data.appet.replace(['good','poor'], [1,0])
kidney_data['pe'] = kidney_data.pe.replace(['yes','no'], [1,0])
kidney_data['ane'] = kidney_data.ane.replace(['yes','no'], [1,0])
kidney_data['classification'] = kidney_data.classification.replace(['ckd','ckd\t','notckd'], [1,1,0])
kidney_data['wc'] = kidney_data.wc.replace(['\t6200','\t8400'], [6200,8400])
kidney_data = kidney_data.dropna(axis=0)
kidney_data.isnull().sum()


# - The reason why I did a separate interpolation on red blood cell and pus cell is because the interpolation method failed to address the NaN values in rbc and pc. Hence, I have to specify it again on the next line. 

# ## Now let's look into the pairplot.

# In[ ]:


gg = sns.pairplot(kidney_data, vars =['age', 'bp','bgr', 'bu', 'sc','sod', 'pot', 'hemo'],hue = 'classification')
gg.map_diag(sns.distplot)
gg.add_legend()
gg.fig.suptitle('FacetGrid plot', fontsize = 20)
gg.fig.subplots_adjust(top= 0.9);


# ## What can we observe here?
# - For majority of the attributes, other than age, we do observe similar peaks between patients with chronic kidney disease and healthy individuals. 
# - Like stated before, the trade-off for choosing to retain data is the creation of less well-behaved results. Yes, we do not see distinct characteristics like the previous Pairplot but that does not mean our data is flawed or lacking. Dealing with large data has its own problem and we should not modify the data to suit our needs or hypotheses. 
# 
# ## Let's look into correlation to have a better understanding on our data.

# In[ ]:


corr = kidney_data.corr()
corr.style.background_gradient(cmap='RdBu_r')


# - I will say that an absolute value of more than 0.4 is considered to be significant.
# - It seems like there are a significant negative correlation between rbc, pc and whether the patient has chronic kidney disease. 
# - Even so, I will look into age, red blood cell, pus cell, blood glucose random, serum creatinine, diabetes mellitus, coronary artery disease, blood urea, sodium, pedal edema and anemia. 

# # Correlation between age and whether a patient has chronic kidney disease

# - Firstly, let's look at the distribution.

# In[ ]:


plt.figure(figsize=(70,25))
plt.legend(loc='upper left')
g = sns.countplot(data = kidney_data, x = 'age', hue = 'classification')
g.legend(title = 'Chronic kidney disease patient?', loc='center left', bbox_to_anchor=(0.1, 0.5), ncol=1)
g.tick_params(labelsize=20)
plt.setp(g.get_legend().get_texts(), fontsize='32')
plt.setp(g.get_legend().get_title(), fontsize='42')
g.axes.set_title('Graph of age vs number of patients with chronic kidney disease',fontsize=50)
g.set_xlabel('Count',fontsize=40)
g.set_ylabel("Age",fontsize=40)


# In[ ]:


age_corr = ['age', 'classification']
age_corr1 = kidney_data[age_corr]
age_corr_y = age_corr1[age_corr1['classification'] == 1].groupby(['age']).size().reset_index(name = 'count')
age_corr_y.corr()


# In[ ]:


sns.regplot(data = age_corr_y, x = 'age', y = 'count').set_title("Correlation graph for Age vs chronic kidney disease patient")


# In[ ]:


age_corr_n = age_corr1[age_corr1['classification'] == 0].groupby(['age']).size().reset_index(name = 'count')
age_corr_n.corr()


# In[ ]:


sns.regplot(data = age_corr_n, x = 'age', y = 'count').set_title("Correlation graph for Age vs healthy patient")


# ## What do we observe here?
# - From what we know, we have approximately 150 healthy subjects and 250 chronic kidney disease patients. 
# - There is a weak positive correlation between age and chronic kidney disease patients. We obtained an R value of approximately 0.387 and an R-square value of approximately 0.150. This means that only 15% of variation can be explained by the relationship between the 2 variables. 
# - The [National Kidney Foundation](https://www.kidney.org/news/monthly/wkd_aging) has associated aging with kidney disease, stating that **"more than 50 percent of seniors over the age of 75 are believed to have kidney disease. Kidney disease has also been found to be more prevalent in those over the age of 60 when compared to the rest of the general population."**
# - Although age is one of the factor that can cause chronic kidney disease, I believe that an unhealthy diet/lifestyle is equally as impactful as age. 

# # Correlation between red blood cell and whether the patient has chronic kidney disease
# 
# Our kidneys create an essential hormone called erythropoietin(EPO). EPO are chemical messengers that plays a key role in the production of red blood cell. Patients with chronic kidney disease has low EPO, resulting in low level of red blood cell. This will eventually lead to anemia. 
# - Given that the red blood cell here is a nominal data, we will need to use Chi-square test to calculate correlation. 
# - We will be using 95% confidence interval (95% chance that the confidence interval you calculated contains the true population mean).
#     * The null hypothesis is that they are independent.
#     * The alternative hypothesis is that they are correlated in some way.

# In[ ]:


# Chi-sq test
cont = pd.crosstab(kidney_data["rbc"],kidney_data["classification"])
scipy.stats.chi2_contingency(cont)


# ## What can we say about this?
# - We performed the test and we obtained a p-value < 0.05 and we can reject the hypothesis of independence. There seem to be a correlation between the condition of red blood cell and whether the patient has chronic kidney disease. 

# # Correlation between pus cell and whether a patient has chronic kidney disease
# - Patients with chronic kidney diseases tend to have pus cell in their urine sample. This could mean that they have infection in the kidney.
# - Given that the pus cell here is a nominal data, we will need to use Chi-square test to calculate correlation. 
# - We will be using 95% confidence interval (95% chance that the confidence interval you calculated contains the true population mean).
#     * The null hypothesis is that they are independent.
#     * The alternative hypothesis is that they are correlated in some way.

# In[ ]:


# Chi-sq test
cont = pd.crosstab(kidney_data["pc"],kidney_data["classification"])
scipy.stats.chi2_contingency(cont)


# ## What can we say about this?
# - We performed the test and we obtained a p-value < 0.05 and we can reject the hypothesis of independence. There seem to be a correlation between the condition of pus cell and whether the patient has chronic kidney disease. 

# # Correlation between blood glucose and whether a patient has chronic kidney disease
# - There is a risk of low blood sugar in patients with chronic kidney disease as kidney function declines insulin and if the patient suffers from diabetes, the diabetes medications will remain in the system longer because of decreased kidney clearance.

# In[ ]:


# Measuring blood glucose and chronic kidney disease patient 
bgr_corr = ['bgr', 'classification']
bgr_corr1 = kidney_data[bgr_corr]
bgr_corr1.bgr = bgr_corr1.bgr.round(-1)
bgr_corr_y = bgr_corr1[bgr_corr1['classification'] == 1].groupby(['bgr']).size().reset_index(name = 'count')
bgr_corr_y.corr()


# In[ ]:


sns.regplot(data = bgr_corr_y, x = 'bgr', y = 'count').set_title("Correlation graph for blood glucose vs chronic kidney disease patient")


# In[ ]:


bgr_corr_n = bgr_corr1[bgr_corr1['classification'] == 0].groupby(['bgr']).size().reset_index(name = 'count')
bgr_corr_n.corr()


# In[ ]:


sns.regplot(data = bgr_corr_n, x = 'bgr', y = 'count').set_title("Correlation graph for blood glucose vs healthy patient")


# ## What do we observe here?
# - There is a strong negative correlation in patient with chronic kidney disease while we observe a strong positive correlation in healthy patients. We obtained an R square value of 0.435 and 0.25 respectively in CKD patients and healthy patients. This means that approximately 44% of the variation can be explained by the relationship of the 2 variables (low blood glucose and chronic kidney disease patients). We do observe a positive relationship betweeen blood glucose and healthy patients as expected. 

# # Correlation between blood urea and chronic kidney disease
# - Urea is the principal nitrogenous waste product of metabolism and is generated from protein breakdown.
# - It is eliminated from the body almost exclusively by the kidneys in urine, and measurement of its concentration, first in urine and later in blood, has had clinical application in the assessment of kidney function for well over 150 years.

# In[ ]:


# Measuring blood urea and chronic kidney disease patient 
bu_corr = ['bu', 'classification']
bu_corr1 = kidney_data[bu_corr]
bu_corr1.bu = kidney_data.bu.round(-1)
bu_corr_y = bu_corr1[bu_corr1['classification'] == 1].groupby(['bu']).size().reset_index(name = 'count')
bu_corr_y.corr()


# In[ ]:


sns.regplot(data = bu_corr_y, x = 'bu', y = 'count').set_title('Correlation graph for blood urea vs CKD patient')


# In[ ]:


bu_corr_n = bu_corr1[bu_corr1['classification'] == 0].groupby(['bu']).size().reset_index(name = 'count')
bu_corr_n.corr()


# In[ ]:


sns.regplot(data = bu_corr_n, x = 'bu', y = 'count').set_title('Correlation graph for blood urea vs healthy patient')


# ## What do we see here?
# - This result seems to be completely opposite of what I thought. You see, urea is mostly removed from the urine and those with chronic kidney disease will have issue removing this waste from the urine. As a result, majority of the urea gets retained in the blood, resulting in high blood urea. However, the negative correlation observed here seems to show a different outlook. What I can say about this is that the patients may be well-informed about his or her condition and might be in a controlled diet, resulting in low amount of blood urea.

# # Correlation between sodium and CKD
# - CKD patients are expected to have low sodium level in their blood. As the damaged kidneys were unable to balance the fluid in the body, large amount of fluid get retained and this lowers the amount of sodium in the blood. 

# In[ ]:


# Measuring blood sodium and chronic kidney disease patient 
sod_corr = ['sod', 'classification']
sod_corr1 = kidney_data[sod_corr]
sod_corr_y = sod_corr1[sod_corr1['classification'] == 1].groupby(['sod']).size().reset_index(name = 'count')
sod_corr_y.corr()


# In[ ]:


sns.regplot(data = sod_corr_y, x = 'sod', y = 'count').set_title('Correlation graph for blood sodium vs CKD patient')


# In[ ]:


sod_corr_n = sod_corr1[sod_corr1['classification'] == 0].groupby(['sod']).size().reset_index(name = 'count')
sod_corr_n.corr()


# In[ ]:


sns.regplot(data = sod_corr_n, x = 'sod', y = 'count').set_title('Correlation graph for blood sodium vs healthy patient')


# ## What do we see here?
# - We do not observe significant correlation between blood sodium and CKD. 

# # Correlation between pedal edema and CKD
# - Pedal edema is caused by excess fluid trapped in the body's tissues. Damaged kidney causes fluid retentions and most of this fluid gets trapped in the hands, arms, feet, ankles and legs.
# - Given that pedal edema here is a nominal data, we will need to use Chi-square test to calculate correlation. 
# - We will be using 95% confidence interval (95% chance that the confidence interval you calculated contains the true population mean).
#     * The null hypothesis is that they are independent.
#     * The alternative hypothesis is that they are correlated in some way.

# In[ ]:


# Chi-sq test
cont = pd.crosstab(kidney_data["pe"],kidney_data["classification"])
scipy.stats.chi2_contingency(cont)


# ## What can we say about this?
# - We performed the test and we obtained a p-value < 0.05 and we can reject the hypothesis of independence. There seem to be a correlation between pedal edema and whether the patient has chronic kidney disease. 

# # Correlation between anemia and CKD
# - Anemia happens when there are insufficient red blood cells to carry out their duties. Our kidneys produce an important hormone called erythropoietin (EPO). This hormone tells your body to make red blood cells. For CKD patients, their kidneys cannot make enough EPO. Low EPO levels cause low red blood cell count, resulting in anemia.
# - Given that anemia here is a nominal data, we will need to use Chi-square test to calculate correlation. 
# - We will be using 95% confidence interval (95% chance that the confidence interval you calculated contains the true population mean).
#     * The null hypothesis is that they are independent.
#     * The alternative hypothesis is that they are correlated in some way.

# In[ ]:


# Chi-sq test
cont = pd.crosstab(kidney_data["ane"],kidney_data["classification"])
scipy.stats.chi2_contingency(cont)


# ## What can we say about this?
# - We performed the test and we obtained a p-value < 0.05 and we can reject the hypothesis of independence. There seem to be a correlation between anemia and whether the patient has chronic kidney disease. 

# # Correlation between serum creatinine and CKD
# - Creatinine is a waste product found in the blood during muscle activities. The kidney is involved in removing this waste material out from the body and when the kidney function is compromised, the amount of creatinine remains in the blood will be high. 

# In[ ]:


# Measuring serum creatinine and chronic kidney disease patient 
sc_corr = ['sc', 'classification']
sc_corr1 = kidney_data[sc_corr]
sc_corr1.sc = sc_corr1.sc.round(1)
sc_corr_y = sc_corr1[sc_corr1['classification'] == 1].groupby(['sc']).size().reset_index(name = 'count')
sc_corr_y.corr()


# In[ ]:


sns.regplot(data = sc_corr_y, x = 'sc', y = 'count').set_title('Correlation graph for serum creatinine vs CKD patient')


# In[ ]:


sc_corr_n = sc_corr1[sc_corr1['classification'] == 0].groupby(['sc']).size().reset_index(name = 'count')
sc_corr_n.corr()


# In[ ]:


sns.regplot(data = sc_corr_n, x = 'sc', y = 'count').set_title('Correlation graph for serum creatinine vs CKD patient')


# ## What do we see here?
# - In terms of distribution we definitely see a bigger range in CKD patients than healthy patients. 
# - We are unable to determine correlation here.

# # Correlation between diabetes mellitus and CKD
# - Diabetes is often associated with CKD and for 45% of patients who receive dialysis therapy, diabetes is the primary cause of their kidney failure. 
# - Additionally, moderate to severe CKD is estimated to be found in 15-23% of patients with diabetes.
# - Given that diabetes here is a nominal data, we will need to use Chi-square test to calculate correlation. 
# - We will be using 95% confidence interval (95% chance that the confidence interval you calculated contains the true population mean).
#     * The null hypothesis is that they are independent.
#     * The alternative hypothesis is that they are correlated in some way.

# In[ ]:


# Chi-sq test
cont = pd.crosstab(kidney_data["dm"],kidney_data["classification"])
scipy.stats.chi2_contingency(cont)


# ## What can we say about this?
# - We performed the test and we obtained a p-value < 0.05 and we can reject the hypothesis of independence. There seem to be a correlation between diabetes and whether the patient has chronic kidney disease. 

# # Correlation between coronary artery disease and CKD
# - Coronary artery disease is the leading cause of morbidity and mortality in patients with CKD. 
# - When you have heart disease, your heart may not be able to pump blood the right way causing pressure to build in the main vein connected to your kidneys. This may lead to a blockage and a reduced supply of oxygen rich blood to the kidneys, leading to kidney disease.
# - Given that coronary artery disease here is a nominal data, we will need to use Chi-square test to calculate correlation. 
# - We will be using 95% confidence interval (95% chance that the confidence interval you calculated contains the true population mean).
#     * The null hypothesis is that they are independent.
#     * The alternative hypothesis is that they are correlated in some way.

# In[ ]:


# Chi-sq test
cont = pd.crosstab(kidney_data["cad"],kidney_data["classification"])
scipy.stats.chi2_contingency(cont)


# ## What can we say about this?
# - We performed the test and we obtained a p-value < 0.05 and we can reject the hypothesis of independence. There seem to be a correlation between coronary artery disease and whether the patient has chronic kidney disease. 

# # Summary
# - We observed strong correlation between CKD and the following:
#     * Red blood cell 
#     * Pus Cell
#     * Blood glucose (strong negative correlation)
#     * Blood urea (strong negative correlation)
#     * Pedal edema 
#     * Anemia 
#     * Diabetes
#     * Coronary artery disease
# - I have explained majority of the sightings but there seem to be a weird observation for blood urea as I was expecting a positive correlation there. What I can say about that is that the CKD patients were informed about their issues and restricted their diet in order to reduce excessive nitrogen level in their blood to a safe level.
