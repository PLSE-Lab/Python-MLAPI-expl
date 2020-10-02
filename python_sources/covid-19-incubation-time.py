#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading libraries
import os
import json
import pandas as pd
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import seaborn as sns
style.use("ggplot")


# In[ ]:


# Loading, pre-processing and organizing data
dir = "/kaggle/input/CORD-19-research-challenge"
dirs = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
data = []

for d in dirs:
    for file in tqdm(os.listdir(f"{dir}/{d}/{d}")):
        paper = json.load(open(f"{dir}/{d}/{d}/{file}", "rb"))
        title = paper['metadata']['title']
        try:
            abstract = paper['abstract'][0]['text']
        except:
            abstract = paper['abstract']
        full_text = ""
        for i in range(0,len(paper['body_text'])):
            full_text += (paper['body_text'][i]['text'])
        data.append([title,abstract,full_text])

df = pd.DataFrame(data, columns = ['title','abstract', 'full_text'])


# In[ ]:


# Processing Data
incubation_times = []
incubation = df[df['full_text'].str.contains('incubation')]
incubation_papers = incubation['full_text'].values
for paper in incubation_papers:
    for sentence in paper.split(". "):
        if "incubation" in sentence:
            duration = re.findall(r"(\d{1,2}(\.\d{1,2})? day[s]?)", sentence)
            if duration:
                duration_str = duration[:][0][0].split(" ")[0]
                incubation_times.append(float(duration_str))
incubation_times = np.array(incubation_times) # Converting the list to NP array
incubation_times = incubation_times[(incubation_times < 30)]  # Removing occurences greater than 30 days


# In[ ]:


mean_duration = round(np.mean(pd.DataFrame(incubation_times).values),2)
print(f"The mean incubation time as cited in {df.shape[0]} papers on Covid-19 is {mean_duration} days.\nThere are {incubation_times.size} instances where the incubation time period is discussed in the scientific community.")            
# Plotting data
plt.ylabel("Number of Instances",fontsize=10,fontweight='bold')
plt.xlabel("Incubation Time (Days)", fontsize=10,fontweight='bold')
ax = sns.distplot(incubation_times, kde=False, bins=30, color="g",hist_kws={'alpha':0})
ax.yaxis.grid(False)
ax.xaxis.grid(False)
ax.set_facecolor((1,1,1))

second_ax = ax.twinx()
plt.ylabel("Percentage of Instances",fontsize=10,fontweight='bold')
ax2 = sns.distplot(incubation_times,ax = second_ax,  color="g",hist_kws={'alpha':.2}, kde_kws={'linewidth':2}, bins=30)
plt.xlim(-1, 30)
plt.title(r"$\bf{" + str(incubation_times.size) + "}$"+ " total instances of incubation time-period \n cited in "
        +  r"$\bf{" + str(df.shape[0])+ "}$" + " scientific papers on Covid-19", fontsize=10)
plt.axvline(mean_duration, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(mean_duration*1.1, max_ylim*0.9, "Mean incubation period is " + r"$\bf{" + str(mean_duration) + "}$" 
         + " days" , fontsize=10)
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
ax2.set_facecolor((1,1,1))
out_dir = "/kaggle/output/working/"
plt.savefig('covid-incubation-time.png', dpi=300)
plt.show()


# In[ ]:




