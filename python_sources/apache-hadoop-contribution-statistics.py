#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import altair as alt
alt.renderers.enable('kaggle')


# In[ ]:


def projects_monthly(dataset, from_date, project_list):
    resolved2019 = dataset[dataset["resolved"] > from_date]

    top_resolved_2019 = resolved2019[resolved2019["project"].isin(project_list)]

    monthly_project = top_resolved_2019[["key","project"]].groupby([top_resolved_2019["resolved"].dt.to_period("M"),"project"]).count();
    monthly_project = monthly_project.reset_index().set_index("resolved").to_timestamp().reset_index()
    return monthly_project

plot_height=450
plot_width=600


# In[ ]:


pd.set_option('display.max_rows', 500)
file = "/kaggle/input/apache-software-foundation-contribution-statistics/jira.csv"
issues = pd.read_csv(file,names=["key","id","summary","author","assignee","created","updated","resolved","resolution","status" ])
issues.resolved = pd.to_datetime(issues.resolved)
issues.created = pd.to_datetime(issues.created)

issues["project"] = issues.key.apply(lambda key: key.split("-")[0])

hadoop_issues = issues[issues["project"].isin(["HDDS","HDFS","MAPREDUCE","YARN","SUBMARINE","HADOOP"])]

resolved = hadoop_issues[hadoop_issues["resolved"].notnull()]
resolved = resolved[resolved["status"].isin(["Closed","Done","Resolved"])]
resolved = resolved[resolved["resolution"].isin(["Done","Fixed","Resolved","Implemented","Information Provided","Workaround","Delivered"])]


# ## Hadoop resolved issues per subprojects in last two years (normalized)

# In[ ]:


show = resolved[resolved["resolved"] > "2018-01-01"]
show = show.groupby([resolved["resolved"].dt.to_period("M"),"project"]).count()
del show["resolved"]
show = show.reset_index().set_index("resolved").to_timestamp().reset_index()

projects_monthly(resolved, "2018-01-01",["HDFS","HDDS","YARN","MAPREDUCE","HADOOP","SUBMARINE"])

alt.Chart(show).mark_area().encode(
    x=alt.X("resolved:T",axis=alt.Axis(title="month")),
    y=alt.Y("key:Q",stack="normalize",axis=alt.Axis(title='# of resolved issues')),
    color="project:N"
).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# ## Hadoop resolved issues per subprojects in last two years

# In[ ]:


show = resolved[resolved["resolved"] > "2018-01-01"]
show = show.groupby([resolved["resolved"].dt.to_period("M"),"project"]).count()
del show["resolved"]
show = show.reset_index().set_index("resolved").to_timestamp().reset_index()

alt.Chart(show).mark_area().encode(
    x=alt.X("resolved:T",axis=alt.Axis(title="month")),
    y=alt.Y("key:Q",axis=alt.Axis(title='# of resolved issues')),
    color="project:N"
).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# ## Resolved issues all time (monthly)

# In[ ]:


show = resolved[["resolved","key"]].groupby([resolved["resolved"].dt.to_period("M")]).count()
del show["resolved"]
show = show.reset_index().set_index("resolved").to_timestamp().reset_index()
alt.Chart(show).mark_bar().encode(
    x=alt.X("resolved:T",axis=alt.Axis(title="month")),
    y=alt.Y("key:Q",axis=alt.Axis(title='# of resolved issues'))
).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# ## Number of unique contributors

# In[ ]:


show = resolved[resolved["assignee"].notnull()]
# hadoop_resolved
show = show[["resolved","assignee","key","project"]].groupby([show["resolved"].dt.to_period("Q"),"assignee","project"]).count()
del show["resolved"]
del show["key"]
assignee_per_month = show.reset_index().groupby(["resolved","project"]).count().reset_index().set_index("resolved").to_timestamp().reset_index()
# assignee_per_month
alt.Chart(assignee_per_month).mark_bar().encode(
    x=alt.X('resolved:N',axis=alt.Axis(title='Time')),
    y=alt.Y("assignee:Q",axis=alt.Axis(title='# unique contributors')),
    color="project:N"
).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# ## Top assignees of the resolved jiras

# In[ ]:


show = resolved[resolved["assignee"].notnull()]
# hadoop_resolved
contributions = show[["resolved","assignee","key"]].groupby(["assignee"]).count().sort_values("key",ascending=False)
# del show["resolved"]
# del show["key"]
contributions.head(20)


# In[ ]:



top_contributors = contributions.head(15).index.to_list()
show = show[show["assignee"].isin(top_contributors)]

show = show[["resolved","assignee","key"]].groupby([show["resolved"].dt.to_period("Q"),"assignee"]).count().sort_values("key",ascending=False)
del show["resolved"]
show = show.reset_index().set_index("resolved").to_timestamp().reset_index()
show
alt.Chart(show).mark_bar().encode(
    x=alt.X('resolved:N',axis=alt.Axis(title='Time')),
    y=alt.Y("key:Q",axis=alt.Axis(title='# resolved issues')),
    color="assignee:N"
).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# ## Long-term contributors (time between last and first contribution)

# In[ ]:


#contribution timeline (time between first and last resolved jira)
result = resolved[resolved["assignee"].notnull()].groupby(["assignee"]).agg({"resolved":[np.min,np.max]})
result["contribution_time"] = result[('resolved', 'amax')] - result[('resolved', 'amin')]
result = result.drop("resolved", axis=1)
result = result.sort_values("contribution_time", ascending=False).dropna().head(30).reset_index()
result.columns = result.columns.get_level_values(0)

result["years"] = result["contribution_time"].dt.days / 365
del result["contribution_time"]
result.columns
bars = alt.Chart(result).mark_bar().encode(
    x='years:Q',
    y=alt.Y("assignee:N",sort=None)
)

bars.properties(height=900)


# In[ ]:



resolved_stat = resolved[["project","resolved"]]    .groupby("project")    .count()    .sort_values("resolved", ascending=False)
resolved_stat

issue_stat = issues[["project","key"]]    .groupby("project")    .count()    .sort_values("key", ascending=False)    .rename(columns={"key":"created"})


contributor_stat = resolved[resolved["assignee"].notnull()]    .groupby(["project","assignee"])    .count().reset_index()
contributor_stat = contributor_stat[["project","assignee"]].groupby("project").count().sort_values("assignee",ascending=False)

subprojects = contributor_stat.join(resolved_stat).join(issue_stat)
subprojects


# In[ ]:


points = alt.Chart(subprojects.reset_index()).mark_circle().encode(
    alt.X('resolved', scale=alt.Scale(type="log"),axis=alt.Axis(title="Number of resolved JIRAs")),
    alt.Y('assignee', scale=alt.Scale(type="log"),axis=alt.Axis(title="Number of unique conributors")),
    color='project'
)
points
text = points.mark_text(
    align='right',
    baseline='middle',
    dx=-4,
    fontSize=12,
).encode(
    text='project',
    size = alt.NumericFieldDefWithCondition(type="quantitative")
)
(points + text).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# In[ ]:


points = alt.Chart(subprojects.reset_index()).mark_circle().encode(
    alt.X('resolved', scale=alt.Scale(type="log")),
    alt.Y('created', scale=alt.Scale(type="log")),
    color='project',
    size=alt.NumericFieldDefWithCondition(type="quantitative",field="assignee",scale=alt.Scale(type="log"))
)
text = points.mark_text(
    align='right',
    baseline='middle',
    dx=-20,
    fontSize=12,
).encode(
    text='project',
    size = alt.NumericFieldDefWithCondition(type="quantitative")
)
(points + text).configure_view(
       height=plot_height,
       width=plot_width,  
    )


# In[ ]:




