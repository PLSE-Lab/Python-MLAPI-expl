#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import altair as alt
alt.renderers.enable('kaggle')


# In[ ]:



pd.set_option('display.max_rows', 500)
file = "/kaggle/input/apache-software-foundation-contribution-statistics/jira.csv"
issues = pd.read_csv(file,names=["key","id","summary","author","assignee","created","updated","resolved","resolution","status" ])
issues.resolved = pd.to_datetime(issues.resolved)
issues.created = pd.to_datetime(issues.created)

issues["project"] = issues.key.apply(lambda key: key.split("-")[0])


# In[ ]:


resolved = issues[issues["resolved"].notnull()]
resolved = resolved[resolved["status"].isin(["Closed","Done","Resolved"])]
resolved = resolved[resolved["resolution"].isin(["Done","Fixed","Resolved","Implemented","Information Provided","Workaround","Delivered"])]

resolved_hadoop_combined = resolved.copy()
resolved_hadoop_combined.loc[resolved_hadoop_combined["project"].isin(["HDDS","HDFS","MAPREDUCE","YARN","SUBMARINE"]),"project"] = "HADOOP"


# In[ ]:


def projects_monthly(dataset, from_date, project_list):
    resolved2019 = dataset[dataset["resolved"] > from_date]

    top_resolved_2019 = resolved2019[resolved2019["project"].isin(project_list)]

    monthly_project = top_resolved_2019[["key","project"]].groupby([top_resolved_2019["resolved"].dt.to_period("M"),"project"]).count();
    monthly_project = monthly_project.reset_index().set_index("resolved").to_timestamp().reset_index()
    return monthly_project
    
    


# ## Users with the most contribution (resolved Jira)

# In[ ]:


#Users with the most contribution
user_contrib = resolved[["key","assignee"]].groupby(["assignee"]).count().sort_values("key", ascending=False)
user_contrib.head(40).reset_index()


# In[ ]:


#Users who contributed to the most project
user_project = issues[["key","assignee","project"]].groupby(["project","assignee"]).count().sort_values("key", ascending=False).reset_index()
user_project.groupby("assignee").count().sort_values("key", ascending=False).head(30)


# In[ ]:


#contribution timeline (time between first and last resolved jira)
by_assignee = issues.groupby("assignee")
result = by_assignee.agg({"resolved":[np.min,np.max]})
result.columns
result["contribution_time"] = result[('resolved', 'amax')] - result[('resolved', 'amin')]
result = result.drop("resolved", axis=1)
result.sort_values("contribution_time", ascending=False).dropna().head(100)


# ## Number of resolved (Jira) issues per month

# In[ ]:


monthly = resolved[["key","resolved"]].groupby([resolved["resolved"].dt.to_period("M")]).count();
del monthly["resolved"]
monthly = monthly.to_timestamp().reset_index()
alt.Chart(monthly).mark_line().encode(
    x='resolved:T',
    y='key:Q'
).configure_view(
    height=400,
    width=800,
)


# ## Flink vs Spark in the last 2 years

# In[ ]:


show = projects_monthly(resolved, "2017-01-01",["FLINK","SPARK"])

alt.Chart(show).mark_area().encode(
    x=alt.X("resolved:T",axis=alt.Axis(title="month")),
    y=alt.Y("key:Q",axis=alt.Axis(title='# of resolved issues')),
    color="project:N"
)


# ## Top projects in 2019 (except INFRA)

# In[ ]:


not_infra = resolved[resolved["project"] != "INFRA"]
top_projects_2019 = not_infra[not_infra["resolved"] > "2019-01-01"][["project","resolved"]]    .groupby("project")    .count()    .sort_values("resolved", ascending=False)    .reset_index()    .head(10)
top_projects_2019


# # Top projects in 2019
# 
# Hadoop subprojects are combined

# In[ ]:


resolved_hadoop_combined = resolved.copy()
resolved_hadoop_combined.loc[resolved_hadoop_combined["project"].isin(["HDDS","HDFS","MAPREDUCE","YARN","SUBMARINE"]),"project"] = "HADOOP"
not_infra = resolved_hadoop_combined[resolved_hadoop_combined["project"] != "INFRA"]

top_projects_2019_hadoop_combined = not_infra[not_infra["resolved"] > "2019-01-01"][["project","resolved"]]    .groupby("project")    .count()    .sort_values("resolved", ascending=False)    .reset_index()    .head(10)
top_projects_2019_hadoop_combined


# In[ ]:


hadoop_combined = issues.copy()
hadoop_combined.loc[hadoop_combined["project"].isin(["HDDS","HDFS","MAPREDUCE","YARN","SUBMARINE"]),"project"] = "HADOOP"

resolved_hadoop_combined = resolved.copy()
resolved_hadoop_combined.loc[resolved_hadoop_combined["project"].isin(["HDDS","HDFS","MAPREDUCE","YARN","SUBMARINE"]),"project"] = "HADOOP"

resolved_stat = resolved_h doop_combined[["project","resolved"]]    .groupby("project")    .count()    .sort_values("resolved", ascending=False)
resolved_stat

issue_stat = hadoop_combined[["project","key"]]    .groupby("project")    .count()    .sort_values("key", ascending=False)    .rename(columns={"key":"created"})


contributor_stat = resolved_hadoop_combined[resolved_hadoop_combined["assignee"].notnull()]    .groupby(["project","assignee"])    .count().reset_index()
contributor_stat = contributor_stat[["project","assignee"]].groupby("project").count().sort_values("assignee",ascending=False)

projects = contributor_stat.join(resolved_stat).join(issue_stat).head(15)


# In[ ]:





# In[ ]:


projects = contributor_stat.join(resolved_stat).join(issue_stat).head(20)
points = alt.Chart(projects.reset_index()).mark_circle().encode(
    alt.X('resolved', scale=alt.Scale(type="log"),axis=alt.Axis(title="Number of resolved JIRAs")),
    alt.Y('assignee', scale=alt.Scale(type="log"),axis=alt.Axis(title="Number of unique conributors")),
    color='project'
)
points
text = points.mark_text(
    align='left',
    baseline='middle',
    dx=4,
    fontSize=16,
).encode(
    text='project',
    size = alt.NumericFieldDefWithCondition(type="quantitative")
)
(points + text).configure_view(
    height=768,
    width=1024,
)


# In[ ]:




