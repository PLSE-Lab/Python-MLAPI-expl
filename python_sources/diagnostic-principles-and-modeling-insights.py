#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import pandas as pd
import numpy  as np
import plotly.express    as px
import matplotlib.pyplot as plt
from sklearn.utils       import resample

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import roc_auc_score, accuracy_score
from IPython.display         import YouTubeVideo


# # What is Melanoma?
# > A quick overview

# In[ ]:


YouTubeVideo('mkYBxfKDyv0', width=800, height=450)


# > Prevalence and Mortality Rate
# - Between 2 to 3 million non-melanoma skin cancers and 132,000 melanoma skin cancers occur globally each year.
# - Melanoma is the deadliest form of skin cancer, although far less prevalent than non-melanoma skin cancers.
# - The incidence of melanoma has more than doubled in the last 30 years.
# - Caught early, we can have as much as 98% treatment success, but this rate drops dramatically as the cancer progresses.

# ![Melanoma Treatment Success by stage](https://www.curemelanoma.org/assets/Uploads/_resampled/ResizedImageWzQwMCwyOTZd/MRA18574-5YearSurviveFig-V1-Hex.png)

# > Risk Factors
# - Family history
# - Personal history
# - Light hair color
# - Light eye color
# - High freckle density
# - Imunosupression
# - Multiple moles

# > Symptons
# - Bleeding
# - Itching
# - Scabbing

# > Examination guidelines - when thinking about data augmentation for the images, make sure to bear these guidelines in mind!
# - (A) Assymetry
# - (B) Borders
# - (C) Color
# - (D) Diameter
# - (E) Elevation and Evolution

# In[ ]:


YouTubeVideo('hXYd0WRhzN4', width=800, height=450)


# # Exploratory Data Analysis

# In[ ]:


# Read files
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df  = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')


# ### How many images do we have per patient?

# In[ ]:


print('Training set contains {} images of {} unique patients, resulting in a ratio of {} images per patient'.format(train_df.shape[0],
                                                                                                                     train_df.patient_id.nunique(),
                                                                                                                     round(train_df.shape[0] / train_df.patient_id.nunique(),2)
                                                                                                                    ))

print('Testing  set contains {} images of {}  unique patients, resulting in a ratio of {} images per patient'.format(test_df.shape[0],
                                                                                                                 test_df.patient_id.nunique(),
                                                                                                                 round(test_df.shape[0] / test_df.patient_id.nunique(),2)
                                                                                                                ))


# ### Do we have missing data?

# > Yes, in both training and testing sets.
# 
# > For the training set, missingness occurs mostly in the `diagnosis` column, but also present in the `age_approx` and `anatom_site_general_challenge`columns.
# Note that for the `diagnosis` column, I'm considering the value 'unknown' as missingness.

# In[ ]:


# Encode 'unknowns' as NaNs
train_df['diagnosis'] = train_df.diagnosis.apply(lambda x: np.nan if x == 'unknown' else x)

labels_df = pd.DataFrame(train_df.benign_malignant.value_counts()).reset_index()
labels_df.columns = ['Label','Count']

# Create dataframe counting NaN values per column
nan_df = pd.DataFrame(train_df.isna().sum()).reset_index()
nan_df.columns  = ['Column', 'NaN_Count']
nan_df['NaN_Count'] = nan_df['NaN_Count'].astype('int')
nan_df['NaN_%'] = round(nan_df['NaN_Count']/train_df.shape[0] * 100,1)
nan_df['Type']  = 'Missingness'
nan_df.sort_values('NaN_%', inplace=True)


# Add completeness
for i in range(nan_df.shape[0]):
    complete_df = pd.DataFrame([nan_df.loc[i,'Column'],train_df.shape[0] - nan_df.loc[i,'NaN_Count'],100 - nan_df.loc[i,'NaN_%'], 'Completeness']).T
    complete_df.columns  = ['Column','NaN_Count','NaN_%','Type']
    complete_df['NaN_%'] = complete_df['NaN_%'].astype('int')
    complete_df['NaN_Count'] = complete_df['NaN_Count'].astype('int')
    nan_df = nan_df.append(complete_df, sort=True)
    
    
# Missingness Plot
fig = px.bar(nan_df,
             x='Column',
             y='NaN_%',
             title='Missingness on the Training Set',
             color='Type',
             template='plotly_dark',
             opacity = 0.6,
             color_discrete_sequence=['#dbdbdb','#38cae0'])

fig.update_xaxes(title='Column Name')
fig.update_yaxes(title='NaN Percentage')
fig.show()


# In[ ]:


# Count NaNs
train_df.isnull().sum()


# > For the testing set, we only have missing values in the `anatom_site_general_challenge` column.

# In[ ]:


labels_df = pd.DataFrame(train_df.benign_malignant.value_counts()).reset_index()
labels_df.columns = ['Label','Count']

# Create dataframe counting NaN values per column
nan_df = pd.DataFrame(test_df.isna().sum()).reset_index()
nan_df.columns  = ['Column', 'NaN_Count']
nan_df['NaN_Count'] = nan_df['NaN_Count'].astype('int')
nan_df['NaN_%'] = round(nan_df['NaN_Count']/test_df.shape[0] * 100,1)
nan_df['Type']  = 'Missingness'
nan_df.sort_values('NaN_%', inplace=True)


# Add completeness
for i in range(nan_df.shape[0]):
    complete_df = pd.DataFrame([nan_df.loc[i,'Column'],test_df.shape[0] - nan_df.loc[i,'NaN_Count'],100 - nan_df.loc[i,'NaN_%'], 'Completeness']).T
    complete_df.columns  = ['Column','NaN_Count','NaN_%','Type']
    complete_df['NaN_%'] = complete_df['NaN_%'].astype('int')
    complete_df['NaN_Count'] = complete_df['NaN_Count'].astype('int')
    nan_df = nan_df.append(complete_df, sort=True)
    
    
# Missingness Plot
fig = px.bar(nan_df,
             x='Column',
             y='NaN_%',
             title='Missingness on the Testing Set',
             color='Type',
             template='plotly_dark',
             opacity = 0.6,
             color_discrete_sequence=['#dbdbdb','#38cae0'])

fig.update_xaxes(title='Column Name')
fig.update_yaxes(title='NaN Percentage')
fig.show()


# In[ ]:


# Count NaNs
test_df.isnull().sum()


# ### How are the labels distributed?

# > Very imbalanced, 1.8% (or 584 images) belong to the positive class.

# In[ ]:


# Summarise data
count_df = labels_df.iloc[::-1]

# Create annotations
annotations = [dict(
            y=count_df.loc[i,'Label'],
            x=count_df.loc[i,'Count'] + 1000,
            text=str(round(count_df.loc[i,'Count']/train_df.shape[0]*100,1))+'%',
            font=dict(
            size=14,
            color="#000000"
            ),
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            bgcolor="#ffffff",
            opacity=0.95,
            showarrow=False,
        ) for i in range(count_df.shape[0])]



fig = px.bar(labels_df,
             y = 'Label',
             x = 'Count',
             title       = 'Label Distribution',
             template    = 'plotly_dark',
             orientation = 'h',
             opacity     = 0.7,
             color       = 'Label',
             color_discrete_sequence = ['#38cae0','#d1324d'] 
            )


fig.update_layout(showlegend=False, annotations = annotations)
fig.show()


# ### How much predictive power does the age, anatomical site and gender carry?

# > They carry moderate predictive power.. a dull logistic regression with only those three variables was able to achieve **0.66 AUROC**.
# 
# > From inspecting the regression coefficients, it is certainly worth considering all variables when creating your own validation set, as they carry some predictive power.

# In[ ]:


fig = px.histogram(train_df,
             x     = 'age_approx',
             color = 'target',
             color_discrete_sequence = ['#38cae0','#d1324d'],
             barnorm  = 'fraction',
             template = 'plotly_dark',
             opacity  = 0.7,
             title    = 'Impact of Age across Diagnosis'
            )

fig.update_xaxes(title = 'Approximated Age', tickvals = list(range(0,91,5)))
fig.update_yaxes(title = 'Percentage of class total')
fig.show()


# > From the plot, we can see that age is correlated with melanoma, this is in line with scientific findings.

# In[ ]:


parallel_df = train_df.copy()

undersampled_df = pd.concat([parallel_df.query("target == 1"),resample(parallel_df.query("target == 0"),
                                                                       replace   = False,
                                                                       n_samples = 584,
                                                                       random_state = 451)
                            ],axis=0)


keep_list = ['sex','age_approx','anatom_site_general_challenge','target']
fig = px.parallel_categories(undersampled_df[keep_list],
                              color="target",
                              template='plotly_dark',
                              labels={"age_approx": "Approximate Age","sex": "Sex", 'anatom_site_general_challenge':'Anatomical Site','target':'Melanoma'},
                              color_continuous_scale=['#dbdbdb','#38cae0'],
                              title='Categorical Flow'
                             )

fig.update_layout(showlegend=False)
fig.show()


# > A few things we can observe:
# - Sex doesn't seem to play a major role, males appear to be more likely to develop melanoma;
# - Younger age groups seem almost unaffected, but have fewer observations as well;
# - Some anatomical sites are more likely to develop this type of cancer, such Head/Neck
# - Oral/genital and Pals/Soles have very little observations to infer anything.

# In[ ]:


def prepare_dataframe(df):
    df['sex'] = np.where(df['sex'] == 'female',1,0)
    df = pd.concat([df.drop('anatom_site_general_challenge',axis=1), pd.get_dummies(df['anatom_site_general_challenge'])],axis=1)
    df = df.drop(['benign_malignant','image_name','patient_id','diagnosis'],axis=1)
    df.loc[df['age_approx'].isnull(),'age_approx'] = df['age_approx'].median()
    
    return(df)

def evaluate_predictions(preds, test_labels):
    '''
    Evaluate Predictions Function
    Returns accuracy and auc of the model
    '''
    auroc = roc_auc_score(test_labels.astype('uint8'), preds)
    accur = accuracy_score(test_labels.astype('uint8'), preds >= 0.5)
    print('Accuracy: ' + str(accur))
    print('AUC: ' + str(auroc))


# In[ ]:


# Split data
train, probe = train_test_split(prepare_dataframe(train_df),
                                test_size = 0.3,
                                stratify = train_df['target'],
                                random_state = 451
                               )

train_y = train.pop('target')
probe_y = probe.pop('target')


# In[ ]:


logit_model = LogisticRegression(random_state=451, solver='lbfgs', max_iter=1000)
logit_model.fit(train, train_y)

logit_preds = logit_model.predict_proba(probe)
evaluate_predictions(logit_preds[:,1], probe_y)


# From these three variables alone we are able to achieve **0.66 AUROC!**
# 
# Now let's see what we can learn from the coeficients.

# In[ ]:


fig = px.bar(y = logit_model.coef_.tolist()[0],
       x = probe.columns.tolist(),
       template = 'plotly_dark',
       title = 'Logistic Regression Coefficient Values',
       color = logit_model.coef_.tolist()[0],
       color_continuous_scale = ['#d1285b','#28b5d1'],
       opacity = 0.7
      )

fig.update_yaxes(title = 'Coefficient Value')
fig.update_xaxes(title = 'Variable Name')
fig.show()


# > We can note a few things:
# - Metadata was very usefull for melanoma classification;
# - Palms/Soles and Oral/Genital may appear very relevant, but have very little data points and little statistical significance.
# - Lower extremity and Torso are less likely to develop melanoma, since they are not as exposed as other regions, though it is not impossible;
# - Head/Neck carries a moderate likelyhood of melanoma with statiscal significance.
# - Age seems small, but this is due to unit of input - the older, the more likely to develop melanoma (this is supported scientifically as well)

# ### What does the Images look like?

# #### Benign class

# In[ ]:


def plot_multiple_images(image_dataframe, rows = 4, columns = 4, figsize = (16, 20), resize=(1024,1024), preprocessing=None, label = 0):
    '''
    Plots Multiple Images
    Reads, resizes, applies preprocessing if desired and plots multiple images from a given dataframe
    '''
    query_string    = 'target == {}'.format(label)
    image_dataframe = image_dataframe.query(query_string).reset_index(drop=True)
    fig = plt.figure(figsize=figsize)
    ax  = []
    base_path = '../input/siim-isic-melanoma-classification/jpeg/train/'
    
    for i in range(rows * columns):
        img = plt.imread(base_path + image_dataframe.loc[i,'image_name'] + '.jpg')
        img = cv2.resize(img, resize)
        
        if preprocessing:
            img = preprocessing(img)
        
        ax.append(fig.add_subplot(rows, columns, i+1) )
        plot_title = "Image {}: {}".format(str(i+1), 'Benign' if label == 0 else 'Malignant') 
        ax[-1].set_title(plot_title)
        plt.imshow(img, alpha=1, cmap='gray')
    
    plt.show()


plot_multiple_images(train_df, label = 0)


# #### Malignant class

# In[ ]:


plot_multiple_images(train_df, label = 1)


# > These images are pretty well behaved.
# From my field experience, I've had contact with many dermatologists, and seldom is the case where the acquisition protocol is so well established. Usually the images are far more spontaneous, with considerably differences regarding brightness, distance, camera type and so on.
# 
# > If you are considering external datasets, which I recommend given the small amount of images, I suggest having a closer look to how the images were acquired.

# # Closing Remarks

# > External datasets may be crucial in order to reach a reasonable amount of **Malignant cases**, note however, that some datasets don't follow the same image acquisition protocol as this one, so I suggest ISIC datasets from previous years.
# 
# > **Age and anatomical site** are very usefull and are certainly advised to be considered when making your splits and models.
# 
# > Many obervations are from the same patient, and thus the patient_id might be useful for good CV splits as well.
# 
# > Images show a moderate variation, due to skin type, hair, locale , which could mean that the combo of **segmentation + classification** could be usefull for this competition.

# ## References

# - https://www.who.int/news-room/q-a-detail/ultraviolet-(uv)-radiation-and-skin-cancer
# - https://www.curemelanoma.org/
# - https://www.wcrf.org/dietandcancer/cancer-trends/skin-cancer-statistics

# ## Recommended resources

# External datasets from ISIC:
# - [ISIC 2019](https://www.kaggle.com/andrewmvd/isic-2019); (Note that the 2019 version contains both 2018 and 2017 datasets as well)
# - [ISIC 2018](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000);
# - [ISIC 2017](https://www.kaggle.com/wanderdust/skin-lesion-analysis-toward-melanoma-detection).
# 
# Also suggest looking at this [thread](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154296) for some info on the competition dataset.
# There is some overlap with these images and the competition training set, however, as we have only 584 images of the positive class, gathering relevant external data will certainly be crucial for this competition.

# Now let's kick some cancer ass.
# 
# If you have any questions, let me know!
