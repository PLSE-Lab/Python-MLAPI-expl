#!/usr/bin/env python
# coding: utf-8

# ### Winning solutions of kaggle competitions: ###
# 
# Awesome work from [SRK](https://www.kaggle.com/sudalairajkumar).
# 
# Generally at the end of every Kaggle competition, winners and other toppers share their solutions in the discussion forum. But it is a tedious task to search for the competitions and then for the solutions of these competitions when we need them. I always wanted to get the links to the solutions of all past Kaggle competitions at one place. I thought it would be a very good reference point many a times. Thanks to Kaggle team for the Meta Kaggle dataset, now I am able to get them in one single place through this notebook .
# 
# Thanks a lot to [@sban](https://www.kaggle.com/shivamb) for this [wonderful kernel](https://www.kaggle.com/shivamb/data-science-glossary-on-kaggle/notebook) without which I wouldn't have got this notebook.
# 
# I have inlcuded only "Featured" & "Research" competitions and the competitions are ordered based on recency.
# 
# Have fun and Happy Kaggling.! 
# 
# P.S : There are few outlier discussions (which are not actually solutions) too ;)
# 

# In[ ]:


import numpy as np
import pandas as pd
from IPython.core.display import HTML

import warnings
warnings.filterwarnings("ignore")

data_path = "../input/"

competitions_df = pd.read_csv(data_path + "Competitions.csv")
comps_to_use = ["Featured", "Research", "Recruitment"]
competitions_df = competitions_df[competitions_df["HostSegmentTitle"].isin(comps_to_use)]
competitions_df["EnabledDate"] = pd.to_datetime(competitions_df["EnabledDate"], format="%m/%d/%Y %I:%M:%S %p")
competitions_df = competitions_df.sort_values(by="EnabledDate", ascending=False).reset_index(drop=True)
competitions_df.head()

forum_topics_df = pd.read_csv(data_path + "ForumTopics.csv")

def check_solution(topic):
    is_solution = False
    to_exclude = ["?", "submit", "why", "what", "resolution", "benchmark"]
    if "solution" in topic.lower():
        is_solution = True
        for exc in to_exclude:
            if exc in topic.lower():
                is_solution = False
    return is_solution

def get_discussion_results(forum_id, n):
    results_df = forum_topics_df[forum_topics_df["ForumId"]==forum_id]
    results_df["is_solution"] = results_df["Title"].apply(lambda x: check_solution(str(x)))
    results_df = results_df[results_df["is_solution"] == 1]
    results_df = results_df.sort_values(by=["Score","TotalMessages"], ascending=False).head(n).reset_index(drop=True)
    return results_df[["Title", "Id", "Score", "TotalMessages", "TotalReplies"]]

def render_html_for_comp(forum_id, comp_name, comp_slug, comp_subtitle, n):
    results_df = get_discussion_results(forum_id, n)
    
    if len(results_df) < 1:
        return
    
    comp_url = "https://www.kaggle.com/c/"+str(comp_slug)
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left;}
                th {
                text-align: left;
                }
            </style>
            <h3><font color="#1768ea"><a href="""+comp_url+""">"""+comp_name+"""</font></h3>
            <p>"""+comp_subtitle+"""</p>
            <table>
            <tr>
                <th><b>S.No</b></th>
                <th><b>Discussion Title</b></th>
                <th><b>Number of upvotes</b></th>
                <th><b>Total Replies</b></th>
            </tr>"""
    
    for i, row in results_df.iterrows():
        url = "https://www.kaggle.com/c/"+str(comp_slug)+"/discussion/"+str(row["Id"])
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  +str(row['Title']) + """</b></a></td>
                    <td>"""+str(row['Score'])+"""</td>
                    <td>"""+str(row['TotalReplies'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))

for ind, comp_row in competitions_df.iterrows():
    render_html_for_comp(comp_row["ForumId"], comp_row["Title"], comp_row["Slug"], comp_row["Subtitle"], 10)


# In[ ]:




