#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas  as pd
import seaborn as sns
import plotly.express    as px
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


# Let's start by reading and inspecting the data file.

# In[ ]:


# Read and inspect data
raw_data = pd.read_csv('../input/violence-against-women-and-girls/makeovermonday-2020w10/violence_data.csv')
raw_data.head()


# In[ ]:


print('Dataset contains data from {} countries'.format(raw_data.Country.nunique()))


# The data was originally provided in the `long` format, we can convert it by using `pivot_table`.
# Lets do this and inspect the contents.

# In[ ]:


raw_survey_df = raw_data.pivot_table(index=['Country','Gender','Demographics Question','Demographics Response'],columns=['Question'], values=['Value'])
raw_survey_df


# Its still a bit chunky, and we can improve by unnesting the indexes and changing the column names.

# In[ ]:


# Reset columns
survey_df = raw_survey_df.T.reset_index(drop=True).T.reset_index()

# Rename columns
survey_df.columns = ['country',
                     'gender',
                     'demographics_question',
                     'demographics_response',
                     'violence_any_reason',
                     'violence_argue',
                     'violence_food',
                     'violence_goingout',
                     'violence_neglect',
                     'violence_sex',
                    ]


# In[ ]:


survey_df


# Note that the columns relate to questions where the respondents were asked if they agreed with the following statements:
# 
# - A husband is justified in hitting or beating his wife for at least one specific reason?
# - A husband is justified in hitting or beating his wife if she argues with him?
# - A husband is justified in hitting or beating his wife if she burns the food?
# - A husband is justified in hitting or beating his wife if she goes out without telling him?
# - A husband is justified in hitting or beating his wife if she neglects the children?
# - A husband is justified in hitting or beating his wife if she refuses to have sex with him?

# # Exploratory Data Analysis

# That's a lot better, now lets query results in order to make an exploratory data analysis acording to `Age` and `Education`.

# In[ ]:


# Examine Violence x gender
fig = px.box(survey_df.query("demographics_question == 'Age'").sort_values('violence_any_reason',ascending=False),
            x      = 'country',
            y      = 'violence_any_reason',
            color  = 'gender',
            title  = '% of Respondents that agree with violence for any surveyed reason across Country and Gender',
            color_discrete_sequence = ['#4a00ba','#00ba82'],
            height = 650
        )

fig.update_xaxes(title='Country')
fig.update_yaxes(title='% Agrees: Violence is justified for any surveyed reason')
fig.show()


# In[ ]:


# Examine Violence x Age group
fig = px.bar(survey_df.query("demographics_question == 'Age'").sort_values('violence_any_reason',ascending=False),
            x      = 'country',
            y      = 'violence_any_reason',
            color = 'demographics_response',
            title  = '% of Violence for any surveyed reason across Country and Age Group ',
            height = 650
        )

fig.update_xaxes(title='Country')
fig.update_yaxes(title='% Agrees: Violence is justified for any surveyed reason')
fig.show()


# In[ ]:


# Examine Correlations
plt.figure(figsize=(10,10))
sns.heatmap(survey_df.iloc[:,4:].corr(),
            square=True,
            linewidths=.5,
            cmap=sns.diverging_palette(10, 220, sep=80, n=7),
            annot=True,
           )
plt.title('Correlation Across Different Violence Questions')
plt.show()

