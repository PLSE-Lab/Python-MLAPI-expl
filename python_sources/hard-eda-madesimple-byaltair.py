#!/usr/bin/env python
# coding: utf-8

# <h3> Welcome all to my EDA of this dataset, with the help of altair you can focus on plotting what you want instead of how you want!!!</h3>
# <h1>Please do upvote!! the content is original as ever and fresh as ever :D</h1>
# 

# In[ ]:




import numpy as np
import pandas as pd 


# In[ ]:


df=pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")


# <h3> here we add altair as alt, whats the altair render script? well to render altair on kaggle notebooks(local machine no need)
# you need to import it, but before that you have to add <a href="https://www.kaggle.com/omegaji/altair-render-script">this</a> as an utility script that's all!!!
# 
# (just click on file and you get the utility script option!!)
# 

# In[ ]:


import altair as alt
import altair_render_script


# <h3> let start with a basic barplot, here we first create a sortdf, what is in this? and what are we doing?
# <ul><li>first take the columns of only the PAID courses</li>
#     <li>then we sort the values according to our number of subscribers in descending order and we select the first <b>20</b></li>
#     <li>now  using alt.chart we input of sortdf give the make_bar for the barchart creation and inside encode we give the x and y</li>
#     <li>x=> course title </li>
#     <li>y=> the number of subscribers 
#     <li> So we are just plotting the top most subscribed course and we also added a toolkit which you can <b>HOVER</b></li>
# </ul>
# </h3>
# <h3><b>Hover for more info,in all graphs I have added the hover option!!</b></h3>

# In[ ]:


sortdf=df[df["is_paid"]==True].sort_values(["num_subscribers"],ascending=False)[:20]
alt.Chart(sortdf).mark_bar().encode(alt.X("course_title"),alt.Y("num_subscribers"),tooltip=["course_title","num_subscribers"]).properties(width=600)
#alt.Chart().mark_bar().encode(x=[1,2,3,4],y=[10,20,30,40])


# <h3> Now we convert the datetime given in the column into days,months,year</h3>

# In[ ]:


from datetime import datetime
def extractdate(x):
    return datetime.strptime(x[:10],"%Y-%m-%d")
df["day"]=df["published_timestamp"].apply(extractdate)
df["day"]=df["day"].apply(lambda x: int(x.day))

df["month"]=df["published_timestamp"].apply(extractdate)
df["month"]=df["month"].apply(lambda x: int(x.month))

df["day_in_year"]=df["published_timestamp"].apply(extractdate)
df["day_in_year"]=df["day_in_year"].apply(lambda x: int(x.timetuple().tm_yday))


df["year"]=df["published_timestamp"].apply(extractdate)
df["year"]=df["year"].apply(lambda x: int(x.year))


# <h3>Now lets do some more advanced!!
# <ul> we create a slider using binding_range ,this slider will be for our years present (2011 to 2017)
#     <li>We create a select year selector which will select the particular year the slider is on in the dataframe</li>
#      
#     <li>we then create a <b> base chart </b> with alt</li>
#      <li>  I have created a courses variable which stores the 4 subject category types</li>
#      <li> we then use the base chart and we select the rows with the first category(source[0]) , group it with months and year and then count the occurences of the course type using count() </li>
#       <li>we then use the selector and transform filter for filtering out the year</li>
#       <li> we then create c1 which will be our line plot using mark_line and we use it on base which we have already configured and filtered</li>
#        <li>we do the same for other courses hence 4 times </li>
#         <li>now we just concat these using concat for the columns and vconcat(vertical concat) for the rows</li>
#       
#      
# </ul>
# </h3>
# <h2> Basically the plot is for showing the amount of courses published of a paritcular category of a particular year and per month</h2>

# In[ ]:



courses=df["subject"].unique()

slider2 = alt.binding_range(min=2011, max=2017, step=1)
select_year= alt.selection_single(name='year', fields=['year'],
                                   bind=slider2, init={'year': 2016})
base = alt.Chart(df[df["subject"]==courses[0]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)

c1=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Buisness Finance Courses made"),tooltip=alt.Tooltip(["course_title"],title="Buisness Finance Courses made"))
base = alt.Chart(df[df["subject"]==courses[1]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)

c2=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Graphic Design Courses made"),tooltip=alt.Tooltip(["course_title"],title="Graphic Design Courses made"))



base = alt.Chart(df[df["subject"]==courses[2]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)

c3=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Musical Instruments Courses made"),tooltip=alt.Tooltip(["course_title"],title="Musical Instruments Courses made"))



base = alt.Chart(df[df["subject"]==courses[3]].groupby(["month","year"]).count().reset_index()).add_selection(select_year).transform_filter(select_year)

c4=base.mark_line(point=True).encode(alt.X("month"), alt.Y("course_title",title="Web Development Courses made"),tooltip=alt.Tooltip(["course_title"],title="Web Development Courses made"))
alt.vconcat(alt.concat(c1,c2,spacing=80),alt.concat(c3,c4,spacing=80),spacing=5)


# <h3>now we create the same selector slider for year</h3>
# <ul><li> create the base like we did above only we are just filtering the subject thats all</li>
#  <li>we want to create barplots, which will tell us the price(total) of that particual level in a particular group of a particular year</li>
#     <h2><li><b>EXAMPLE: In the year 2017 the 3rd month(March) under the Buisness Finance group the total courses prices was 1795</b></li></h2>
# </ul>

# In[ ]:



courses=df["subject"].unique()
slider2 = alt.binding_range(min=2011, max=2017, step=1)
select_year= alt.selection_single(name='year', fields=['year'],
                                   bind=slider2, init={'year': 2016})
base = alt.Chart(df[df["subject"]==courses[0]]).add_selection(select_year).transform_filter(select_year)



a=base.mark_bar(size=10).encode(
    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Buisness Finance Prices over the month")

b=base.mark_bar(size=10).encode(
    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Graph Design Prices over the month")

c=base.mark_bar(size=10).encode(
    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Musical Instrument Prices over the month")

d=base.mark_bar(size=10).encode(
    alt.X("month"),alt.Y("sum(price)"),color="level",tooltip=alt.Tooltip(["sum(price)"],title="Total Price")).properties(title="Web Development Prices over the month")
alt.vconcat(alt.concat(a,b,spacing=20),alt.concat(c,d,spacing=20),spacing=5)


# <h3> we should also plot the top subscribers for FREE courses just as we did for paid courses</h3>

# In[ ]:


sortdf=df[df["is_paid"]==False].sort_values(["num_subscribers"],ascending=False)[:20]
alt.Chart(sortdf).mark_bar().encode(alt.X("course_title"),alt.Y("num_subscribers"),tooltip=["course_title","num_subscribers"]).properties(width=600)


# <h3> NOT ENOUGH BARPLOTS :D</h3>
# we are still gonna plot more :D
# <h3>over here we simply plot barplots with the number of subscribers for different kinds of levels(all,beginner...) and different kinds of groups(buisness,graphic,...)

# In[ ]:


alt.Chart(df).mark_bar().encode(
    column='level',
    x='num_subscribers',
    y='subject' ,tooltip=["num_subscribers"]
).properties(width=220)


# <h3> do the same but instead of number of subscribers we have number of lectures!!</h3>

# In[ ]:


alt.Chart(df).mark_bar().encode(
   column="level",
    x='num_lectures',
    y='subject' ,tooltip=["num_lectures"]
).properties(width=220)


# <h3> NOT ENOUGH BARS NOT ENOUGH SLIDERS!!! :D</h3>
# <ul> Here we use 2 sliders one for year and one for month </ul>
#  <li>create the sliders same way as we did above </li>
#   <li>create the base chart and add the selection and filter functions for year month respectively</li>
#    <li>now create a bar chart (we also filter out and keep the paid courses only) for number of subscribers for the 4 subjects over the years and over the months :D</li>

# In[ ]:


slider = alt.binding_range(min=1, max=12, step=1)
select_month = alt.selection_single(name='month', fields=['month'],
                                   bind=slider, init={'month': 1})

slider2 = alt.binding_range(min=2011, max=2017, step=1)
select_year= alt.selection_single(name='year', fields=['year'],
                                   bind=slider2, init={'year': 2016})
base = alt.Chart(df).add_selection(select_year,select_month).transform_filter(select_year).transform_filter(
    select_month
)




left = base.transform_filter(alt.datum.is_paid==True).encode( 
     y=alt.Y('subject'),
     x=alt.X('sum(num_subscribers)',
            
            title='NumOfSubscribers')
    ,tooltip=["sum(num_subscribers)","subject"]).mark_bar(size=20).properties(title='subscribers Over the month for PAID',height=200)

left


# <h3> we do the same only for non paid(free) courses</h3>

# In[ ]:



right = base.transform_filter(alt.datum.is_paid==False).encode(
     y=alt.Y('subject'),
    x=alt.X('sum(num_subscribers)'),tooltip=["sum(num_subscribers)","subject"]).mark_bar(size=20).properties(title='subscribers Over the month for FREE',height=200)


right


# <h3> Now lets do something with the number of lectures, we create a bar chart with bins(20 is the size(step=21)) for the number of lectures , the y axis is the count of the subjects present in this number of lecture bin,the color depicts the subject notation, also the <b>white</b> strips use is because I added num_lectures to toolkit list, hence it also shows the specific number of lectures that particular strip has</h3>

# In[ ]:




alt.Chart(df).mark_bar(size=10).encode(
    alt.X("num_lectures:Q", bin=alt.Bin(step=21)),
    alt.Y("count(subject)",title="subject count "),
    row='level',color='subject',tooltip=["count(subject)","subject","num_lectures"]
).properties(width=700)


# <h3>we do the same for our course content duration!!!</h3>

# In[ ]:




alt.Chart(df).mark_bar(size=13).encode(
    alt.X("content_duration:Q", bin=alt.Bin(step=2)),
    alt.Y("count(subject)",title="subject count "),
    row='level',color='subject',tooltip=["count(subject)","subject","content_duration"]
).properties(width=700)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




