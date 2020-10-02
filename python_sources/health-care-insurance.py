#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


# In[ ]:


database = ("../input/database.sqlite")
conn = sqlite3.connect(database)


# In[ ]:




tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables


# Total 8 tables. First check how many years of the information.

# In[ ]:


year=pd.read_sql("""SELECT DISTINCT BusinessYear FROM Rate;""", conn)
year


# 

# Three years of information available. The following part is about the sources that provide service across the country, from SourceArea table.

# In[ ]:


table_servicearea = pd.read_sql("""SELECT *
                        FROM servicearea
                        LIMIT 100;""", conn)
table_servicearea


# first let's look at how many states have been included in this dataset.

# In[ ]:


state_in_area=pd.read_sql("""SELECT DISTINCT statecode from servicearea order by statecode;""", conn)
state_in_area


# A little surprise. the data is from 40, not 50 states of the country. 
# Let's look at how many different  issuers(plan provider?), service area across the country.

# In[ ]:


servicearea_issuerid=pd.read_sql("""select statecode, count(distinct(issuerid)) as num_issuer
                                    , count(distinct(serviceareaid)) as num_service
                                    , count(distinct (serviceareaname)) as num_servicename
                                    from servicearea
                                    group by statecode
                                    order by statecode;""", conn)
servicearea_issuerid


# from the results, it seems serviceareaid and serviceareaname are two different definitions. the number of service area id is less than that of service area name. I am not sure if that means certain plan providers offer service in several areas (with name), and multiple service areas (with name) share one service area id.

# Now let's take a look at how many sources across the country.

# In[ ]:


source_type=pd.read_sql("""SELECT distinct(SourceName)
                        from ServiceArea 
                        ; """, conn)
source_type


# Three sources through which data was collected. 

# In[ ]:


# see how the popuar of each source in states
source_popularity=pd.read_sql("""select sourcename, count(distinct(statecode)) 
                            from servicearea group by sourcename; """, conn)
source_popularity


# OPM is the most popular source that was used in 27 states.

# In[ ]:


# see how many services of each source in states
source_count=pd.read_sql("""select sourcename, count(statecode) as service_number
                            from servicearea group by sourcename order by sourcename; """, conn)
source_count


# Although OPM serves 27 states while SERFF's service was in only 20 states, SERFF has the most data input counts but OPM has the least.
# 
# Let's see the distributions in different states.

# In[ ]:


source_state=pd.read_sql("""SELECT StateCode, SourceName, COUNT(SourceName) as num_source 
                        from ServiceArea 
                        group by StateCode, SourceName 
                        order by num_source desc; """, conn)
source_state


# -----------------------------------------------------------------
# Now let's take a look at table 'rate'.

# In[ ]:


table_rate = pd.read_sql("""SELECT *
                        FROM Rate
                        LIMIT 30;""", conn)
table_rate


# First, what does the average rate look like in each state?

# In[ ]:


rate_state=pd.read_sql("""select businessyear, statecode, avg(individualrate) as rate_ave
                        from rate 
                        group by businessyear, statecode
                        order by rate_ave desc;""", conn)
rate_state


# Very surprisingly, average rate in 2014 is very very high in most states. in some top states the number even reached 5 digits... since WY is the top 1, let's look at it more closely.

# In[ ]:


rate_wy_TOP=pd.read_sql("""select individualrate 
                        from rate 
                        where statecode='WY' and businessyear=2014
                        ORDER BY INDIVIDUALRATE DESC
                        limit 1000;""", conn)
rate_wy_TOP


# one million??? is that a real number or something like system bug? let's take a look at other states.

# In[ ]:


rate_AK_TOP=pd.read_sql("""select individualrate 
                        from rate 
                        where statecode='AK' and businessyear=2014
                        ORDER BY INDIVIDUALRATE DESC
                        LIMIT 550;""", conn)
rate_AK_TOP


# So in AK, it seems the same thing, and the next level is $ 1900.
# 
# Also it seems the average in 2016 is much lower, so how about the rate in 2016 in WY?

# In[ ]:


rate_wy_TOP_2016=pd.read_sql("""select individualrate 
                        from rate 
                        where statecode='WY' and businessyear=2016
                        ORDER BY INDIVIDUALRATE DESC
                        LIMIT 1000;""", conn)
rate_wy_TOP_2016


# The million level disappeared. The highest level in 2016 is in the similar range as 2014 secondary level, which is less than $2000.
# 
# Then let's see what planid is associated with the million dollar rate.

# In[ ]:


rate_PLAN_WY=pd.read_sql("""select DISTINCT(PLANID)
                                from rate 
                                WHERE STATECODE='WY' AND INDIVIDUALRATE=999999
                                ;""", conn)
rate_PLAN_WY


# Only four plan ids are responsible for the million dollar rate. How many plan ids are there in WY?

# In[ ]:


rate_PLANS_2014_WY=pd.read_sql("""select DISTINCT(PLANID)
                                from rate 
                                WHERE BUSINESSYEAR=2014 AND STATECODE='WY'
                                ;""", conn)
rate_PLANS_2014_WY


# So definitely there are other plans in WY that charged less.
# 
# Lets get all those million dollar plans in all states in 2014.

# In[ ]:


rate_PLANS_2014_EXP=pd.read_sql("""select  STATECODE, PLANID
                                    from rate 
                                    WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999
                                    GROUP BY PLANID
                                    ORDER BY STATECODE;""", conn)
rate_PLANS_2014_EXP


# These plans have common features: 1) statecode is embedded in the middle of the id. 2) they all end with 0001 or 0002.
# 
# We noticed the high rate disappeared in 2016. So is that because the rate decreased or the plan was canceled?

# In[ ]:


rate_PLANS_2016_2014EXP=pd.read_sql("""SELECT STATECODE, PLANID
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2016 AND 
                                    PLANID IN
                                    (select PLANID
                                    from rate 
                                    WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999
                                    )
                                     ;""", conn)
rate_PLANS_2016_2014EXP


# seems the plans were canceled in 2016. To assure it, try several examples.

# In[ ]:


rate_PLANS_2016=pd.read_sql("""SELECT STATECODE, PLANID
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2016 AND 
                                    PLANID = '74819AK0010001'
                                     ;""", conn)
rate_PLANS_2016


# In[ ]:


rate_PLANS_2016WY=pd.read_sql("""SELECT DISTINCT(PLANID)
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2016 AND 
                                    STATECODE = 'WY' 
                                    AND PLANID NOT IN ( '47731WY0030002', '47731WY0030001','47731WY0020002', '47731WY0020001' ) 
                                     ;""", conn)
rate_PLANS_2016WY


# So now, we are pretty sure that those million dollar plans were canceled in 2016.
# 
# Then next question is what are those plans, instead of just plan id? now we need another table plan attributes.

# In[ ]:


PLANTYPE=pd.read_sql("""SELECT PLANID, PLANTYPE, BenefitPackageId
                        FROM PLANATTRIBUTES
                        WHERE PLANID IN
                                    (select PLANID
                                    from rate 
                                    WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999)
                        ;""", conn)
PLANTYPE


# No record???!!! check WY state since we know them so well.

# In[ ]:


PLANTYPE1=pd.read_sql("""SELECT PLANID, PLANTYPE, BenefitPackageId
                        FROM PLANATTRIBUTES
                        WHERE PLANID IN
                                    (SELECT DISTINCT(PLANID)
                                    FROM RATE
                                    WHERE BUSINESSYEAR=2014 AND 
                                    STATECODE = 'WY' 
                                    AND PLANID NOT IN ( '47731WY0030002', '47731WY0030001','47731WY0020002', '47731WY0020001'))
                        ;""", conn)
PLANTYPE1


# Still no record???!!! need look at the new table carefully

# In[ ]:


PLANID_IN_ATTRI=pd.read_sql("""SELECT DISTINCT (PLANID)
                                FROM planattributes
                                where statecode='WY' AND BUSINESSYEAR=2014;""", conn)
PLANID_IN_ATTRI


# Unbelievable! The same feature in different  tables has different data format! If you notice the last several plan ids, they are the million dollar plans. But in this table, these plan ids have extra "-xx" part!!!
# 
# Then we need try again with the function to extract the part in "rate" table.

# In[ ]:


PLANTYPE_MODIFY=pd.read_sql("""SELECT planid, PLANTYPE, BenefitPackageId, PlanMarketingName, ISSUERID
                                FROM PLANATTRIBUTES
                                WHERE SUBSTR(PLANATTRIBUTES.PLANID,1, 14) IN
                                (select PLANID
                                from rate 
                                WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999)
                                ;""", conn)
PLANTYPE_MODIFY


# Now we got all of them. It seems they are all PPO plans, and are all dental plans? Check it!

# In[ ]:


PLANTYPE_MODIFY=pd.read_sql("""SELECT PLANTYPE, PlanMarketingName
                                FROM PLANATTRIBUTES
                                WHERE SUBSTR(PLANATTRIBUTES.PLANID,1, 14) IN
                                (select PLANID
                                from rate 
                                WHERE BUSINESSYEAR=2014 AND INDIVIDUALRATE=999999)
                                GROUP BY PLANMARKETINGNAME
                                ;""", conn)
PLANTYPE_MODIFY


# So now we know all those million dollar plans in 2014 are all dental PPO plans. for future analysis, probably we need separate dental plan from other insurance plans to make comparison among states.

# First take a look at the average rate from the states.

# In[ ]:


rate_state_reg=pd.read_sql("""select businessyear, statecode, avg(individualrate) as rate_ave
                        from rate 
                        WHERE INDIVIDUALRATE != 999999
                        group by businessyear, statecode
                        order by STATECODE;""", conn)
rate_state_reg


# In[ ]:


rate_state_pivot1=pd.read_sql("""select  statecode, businessyear,avg(individualrate) as rate_ave
                                        from rate 
                                        WHERE businessyear in (2014, 2015, 2016) and INDIVIDUALRATE != 999999
                                        group by businessyear, statecode
                                        ;""", conn)
rate_state_pivot1


# This time, the numbers look reasonable. Let's make it to pivot table

# In[ ]:


rate_state_pivot=pd.read_sql("""select statecode,
                                        SUM(CASE WHEN BusinessYear = 2014 THEN rate_ave END) AS '2014',
                                         SUM(CASE WHEN BusinessYear = 2015 THEN rate_ave  END) AS '2015',
                                         SUM(CASE WHEN BusinessYear = 2016 THEN rate_ave  END) AS '2016'
                                from (select  statecode, businessyear,avg(individualrate) as rate_ave
                                        from rate 
                                        WHERE INDIVIDUALRATE != 999999
                                        group by businessyear, statecode
                                        )
                                group by statecode;""", conn)
rate_state_pivot


# How many plans are dental plans?

# In[ ]:


dental_plan=pd.read_sql("""select statecode, businessyear, count(distinct(planid)) as num_dental
                                from planattributes
                                where dentalonlyplan = 'Yes'
                                group by statecode, businessyear
                                order by statecode;""", conn)
dental_plan


# How many total plan numbers are there?

# In[ ]:


total_plan=pd.read_sql("""select statecode, businessyear, count (distinct (planid)) as total_plan
                                from planattributes
                                group by statecode, businessyear
                                order by statecode;""", conn)
total_plan


# In[ ]:


dental_total_plan=dental_plan.merge(total_plan)
dental_total_plan


# Look at medical insurance rate and dental insurance rate separately.

# In[ ]:


medical_rate=pd.read_sql("""select rate.statecode, rate.businessyear, avg(rate.individualrate) as medical_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='No' 
                            group by rate.statecode, rate.businessyear
                            order by rate.statecode;""", conn)
medical_rate


# In[ ]:


dental_rate=pd.read_sql("""select rate.statecode, rate.businessyear, avg(rate.individualrate) as medicine_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='Yes' 
                            group by rate.statecode, rate.businessyear
                            order by rate.statecode;""", conn)
dental_rate


# Forgot to exclude the million dollar plans...

# In[ ]:


dental_realrate=pd.read_sql("""select rate.statecode, rate.businessyear, avg(rate.individualrate) as dental_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='Yes' and rate.individualrate !=999999 
                            group by rate.statecode, rate.businessyear
                            order by rate.statecode;""", conn)
dental_realrate


# In[ ]:


medical_dental_rate=medical_rate.merge(dental_realrate)
medical_dental_rate


# Does the rate has any relationship with age? First look at how many age groups are there?

# In[ ]:


age_rate=pd.read_sql("""select distinct (age) from rate;""", conn)
age_rate


# In[ ]:


rate_age=pd.read_sql("""select avg(individualrate) as rate, age
                        from rate
                        where individualrate !=999999
                        group by age
                        ;""", conn)
rate_age


# In[ ]:


fig, ax=plt.subplots(figsize=[20, 5])
sns.barplot(x='Age', y='rate', data=rate_age)


# The average rate definietly goes up with age. Is it true for both medical and dental rate?

# In[ ]:


medical_rate_age=pd.read_sql("""select rate.statecode, avg(rate.individualrate) as medical_rate, rate.age
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='No' 
                            group by rate.statecode, rate.age
                            order by rate.statecode;""", conn)
medical_rate_age


# In[ ]:


medical_rate_age=medical_rate_age.pivot(index= 'StateCode', columns= 'Age', values='medical_rate')
medical_rate_age.head()


# In[ ]:


fig, ax=plt.subplots(figsize=[20,10])
sns.heatmap(medical_rate_age)


# It seems for medical insurance, the rate is positively related with age. especially in some states like AK, WY, NJ and NC, the medical rate is higher than other states.

# In[ ]:


dental_realrate_age=pd.read_sql("""select rate.statecode,  avg(rate.individualrate) as dental_rate, rate.age
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.dentalonlyplan='Yes' and rate.individualrate !=999999 
                            group by rate.statecode, rate.age
                            order by rate.statecode;""", conn)
dental_realrate_age


# In[ ]:


dental_realrate_age=dental_realrate_age.pivot(index= 'StateCode', columns= 'Age', values='dental_rate')
dental_realrate_age.head()


# In[ ]:


fig, (axes1, axes2)=plt.subplots(2,1,figsize=[20,20])
sns.heatmap(medical_rate_age, ax=axes1)
sns.heatmap(dental_realrate_age, ax=axes2)


# Surprisingly dental insurance rate has no relationship with age... And some states have super high level, like UT
# Let's take a look at dental plans in UT

# In[ ]:


dental_UT=pd.read_sql("""select rate.planid, rate.individualrate as dental_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.statecode= 'UT' and planattributes.dentalonlyplan='Yes'
                            group by rate.planid
                            order by dental_rate desc;""", conn)
dental_UT


# So besides millions dollar plans, UT has 10,000 dollar dental plans...

# In[ ]:


dental_wy=pd.read_sql("""select rate.planid, rate.individualrate as dental_rate
                            from rate
                            inner join planattributes on rate.planid=substr(planattributes.planid, 1,14)
                            where planattributes.statecode= 'WY' and planattributes.dentalonlyplan='Yes'
                            group by rate.planid
                            ORDER BY dental_rate desc;""", conn)
dental_wy


# So WY state has only million dollar super dental plan. There is no 10,000 dollar dental rate. That's why the average dental rate is so low in WY.

# Keep updating!
