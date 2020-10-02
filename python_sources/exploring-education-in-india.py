#!/usr/bin/env python
# coding: utf-8

# # Exploring Education in India
# 
# I wanted to analyze the education system in India. Particularly, I wanted to investigate the factors of gender and government involvment in affecting the total enrollment of students over time. At first, I wanted to see how India compares to other contries in terms of total enrollment and total spending in education. After getting an overview of how India compares with other countries, it would be helpful to analyze any trends that can be found within different grades. 
# 
# Table of Contents
# -----------------------------------------------------------------------
# 
# 1.  How does student enrollment in India compare with other Asian countries?
#     - Primary enrollment in India and nearby countries
#     - Secondary enrollment in India and nearby countries
# 2. How much does student enrollment correlate with the level of investment in education?
#     - Education expenditure for India and nearby countries
# 3. In India, does the number of enrollment and completion differ based on gender?
#     - Primary enrollment and completion rate in India
#     - Secondary enrollment and completion rate in India
# 4. Conclusion

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_style("white")

# read dataset
country = pd.read_csv('../input/Country.csv')
indicators = pd.read_csv('../input/Indicators.csv')


# ## 1. How does student enrollment in India compare with other Asian countries?
# 
# I wanted to understand student enrollement in India and compare that with other Asian countires. The four other countries were chosen based on their GDP. India has one of the highest GDP in Asia so it would be beneficial to compare it with other countries in the region with a similar GDP. It would not be helpful to compare student enrollment based on exact number of enrolled students because of the large difference in population between the countries. Instead, I looked at the gross enrollment ratio, which might be a better indicator when comparing student enrollment in different countries. 

# In[ ]:


# Primary Gross Enrollment Rate (total)
ind_pri_enrr = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # India
chn_pri_enrr = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # China
jpn_pri_enrr = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # Japan
rus_pri_enrr = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # Russia
idn_pri_enrr = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # Indonesia

# Secondary Gross Enrollment Rate (total)
ind_sec_enrr = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # India
chn_sec_enrr = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # China
jpn_sec_enrr = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # Japan
rus_sec_enrr = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # Russia
idn_sec_enrr = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # Indonesia

sns.set_palette("husl")

fig = plt.figure()
plt.title('Primary, Gross Enrollment Ratio (GER)')
plt.plot(ind_pri_enrr.Year, ind_pri_enrr.Value,  label='India')
plt.plot(chn_pri_enrr.Year, chn_pri_enrr.Value,  label='China')
plt.plot(jpn_pri_enrr.Year, jpn_pri_enrr.Value,  label='Japan')
plt.plot(rus_pri_enrr.Year, rus_pri_enrr.Value,  label='Russia')
plt.plot(idn_pri_enrr.Year, idn_pri_enrr.Value,  label='Indonesia')
plt.ylabel('Gross Enrollment Ratio')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)

fig = plt.figure()
plt.title('Secondary, Gross Enrollment Ratio (GER)')
plt.plot(ind_sec_enrr.Year, ind_sec_enrr.Value,  label='India')
plt.plot(chn_sec_enrr.Year, chn_sec_enrr.Value,  label='China')
plt.plot(jpn_sec_enrr.Year, jpn_sec_enrr.Value,  label='Japan')
plt.plot(rus_sec_enrr.Year, rus_sec_enrr.Value,  label='Russia')
plt.plot(idn_sec_enrr.Year, idn_sec_enrr.Value,  label='Indonesia')
plt.ylabel('Gross Enrollment Ratio')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)


# We can observe from the primary enrollment ratio graph, that India has steady increased in primary enrollment over the years. It appears that most other countries with high GDPs in Asia also have a high primary enrollment ratio. It is important to note that it is possible to go over 100 when it comes to gross enrollment ratio (GER) as we can see with all of the five countries in the graph. The GER is the ratio of the number of students enrolled in school to the number of children in the country of the corresponding age. The GER can be above 100 as a result of students repeating grades and from students enrolling in the grade earlier or later than the normal age for the grade. 
# 
# There is a significant drop in the GER for secondary education from primary education, for three out of the five countries. However, all three of these countries (India, China and Indonesia) are seeing a consistent increase in secondary education GER. There increase is almost the same for the three countries. Japan is consistently at 100 which means that almost all the children of school age are enrolled in school. The one outlier in recent years is Russia, who had a dip around 2003 but has been rising again after 2009. Overall India, along with the other four countries, is consistently increasing the student enrollment ratio.  

# ## 2. How much does student enrollment correlate with the level of investment in education?
# 
# It is generally understood that an increase in government expenditure on education correlates with increased level of student engagement. However, I wanted to explore the extent to which this correlation is true. In particular, I wanted to find if the correlation differs for primary education versus secondary education.

# In[ ]:


ind_edu_exp_US = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # India
chn_edu_exp_US = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # China
jpn_edu_exp_US = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # Japan
rus_edu_exp_US = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # Russia
idn_edu_exp_US = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # Indonesia

# Adjusted savings: education expenditure (% of GNI) 
ind_edu_exp_GNI = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # India
chn_edu_exp_GNI = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # China
jpn_edu_exp_GNI = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # Japan
rus_edu_exp_GNI = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # Russia
idn_edu_exp_GNI = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # Indonesia

fig = plt.figure()
plt.title('Adjusted savings education expenditure (Current US Dollar)')
plt.plot(ind_edu_exp_US.Year, ind_edu_exp_US.Value, 'o-', label='India')
plt.plot(chn_edu_exp_US.Year, chn_edu_exp_US.Value, 'o-', label='China')
plt.plot(jpn_edu_exp_US.Year, jpn_edu_exp_US.Value, 'o-', label='Japan')
plt.plot(rus_edu_exp_US.Year, rus_edu_exp_US.Value, 'o-', label='Russia')
plt.plot(idn_edu_exp_US.Year, idn_edu_exp_US.Value, 'o-', label='Indonesia')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)

fig = plt.figure()
plt.title('Adjusted savings education expenditure as % of Gross National Income')
plt.plot(ind_edu_exp_GNI.Year, ind_edu_exp_GNI.Value, 'o-', label='India')
plt.plot(chn_edu_exp_GNI.Year, chn_edu_exp_GNI.Value, 'o-', label='China')
plt.plot(jpn_edu_exp_GNI.Year, jpn_edu_exp_GNI.Value, 'o-', label='Japan')
plt.plot(rus_edu_exp_GNI.Year, rus_edu_exp_GNI.Value, 'o-', label='Russia')
plt.plot(idn_edu_exp_GNI.Year, idn_edu_exp_GNI.Value, 'o-', label='Indonesia')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)


# In order to account for the difference in GDP of the different countries, I looked at both the total amount of expenditure in education as well as the percent of expenditure in education as a percent of the gross national income. Japan came on top in both of the metrics. From our previous analysis we know that Japan also has consistently been close to 100 percent in gross enrollment rate. When we look at the expenditure in education with the other countries, we can see a correlation between the amount of student enrollment with the level of investment in education. In India, the percent of investment in education jumped between 1995 and 2000. Interestingly, that is the same time period that the GER for India started to increase consistently. From the adjusted savings expenditure graph, we can observe that China had started to increase its expenditure near 1993, which is also the same time that its secondary education GER began to steadily increase. Although there does not seem to be a direct correlation between government expenditure in education and primary student enrollment, there is a definite correlation between government expenditure in education and secondary school student enrollment.

# ## 3. Does the enrollment rate and the completion rate differ based on gender in India?
# 

# In[ ]:


# Primary completion rate 
pri_comp_fm = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.CMPT.FE.ZS')]  # female
pri_comp_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.CMPT.MA.ZS')]  # male 
pri_comp_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.CMPT.ZS')]  # both sexes 

# Lower Secondary completion rate
lo_sec_comp_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.CMPT.LO.ZS')]# both
lo_sec_comp_fe = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.CMPT.LO.FE.ZS')]# female
lo_sec_comp_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.CMPT.LO.MA.ZS')]# male

# Primary Gross enrollment ratio
pri_enrr_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # both SE.PRM.ENRR
pri_enrr_fm = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR.FE')] # female
pri_enrr_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR.MA')] # male

# Secondary gross enrollment ratio
sec_enrr_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR')]# both
sec_enrr_fe = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR.FE')]# female
sec_enrr_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR.MA')]# male

fig = plt.figure()
plt.title('Gross Enrollment Ratio (GER)')
plt.plot(pri_enrr_both.Year, pri_enrr_both.Value, 'bo-', label='Primary education, all')
plt.plot(pri_enrr_fm.Year, pri_enrr_fm.Value, 'go-', label='Primary education, female')
plt.plot(pri_enrr_ma.Year, pri_enrr_ma.Value, 'ro-', label='Primary education, male')

plt.plot(sec_enrr_both.Year, sec_enrr_both.Value, 'b--', label='Secondary education, all')
plt.plot(sec_enrr_fe.Year, sec_enrr_fe.Value, 'g--', label='Secondary education, female')
plt.plot(sec_enrr_ma.Year, sec_enrr_ma.Value, 'r--', label='Secondary education, male')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)

fig = plt.figure()
plt.title('Completion Rate')
plt.plot(pri_comp_both.Year, pri_comp_both.Value, 'bo-', label='Primary, all')
plt.plot(pri_comp_fm.Year, pri_comp_fm.Value, 'go-', label='Primary, female')
plt.plot(pri_comp_ma.Year, pri_comp_ma.Value, 'ro-', label='Primary, male')

plt.plot(lo_sec_comp_both.Year, lo_sec_comp_both.Value, 'ko-', label='Lower Secondary, all')
plt.plot(lo_sec_comp_fe.Year, lo_sec_comp_fe.Value, 'co-', label='Lower secondary, female')
plt.plot(lo_sec_comp_ma.Year, lo_sec_comp_ma.Value, 'yo-', label='Lower secondary, male')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)


# We can see that before 2000, males had a higher GER and completion rate than females. More importantly, the difference between the two were constant before the year 2000. After that we see the gap between female and male GER decrease in both primary and secondary education. Additionally the completion rate for both females and males are currently almost the same. 

# ## Conclusion
# 
# Overall, there is a significant improvement with enrollment rates and completion rates in India in the past two decades. While it seems that more children in India are getting a primary education, they do not stay on for secondary education. We could see a significant drop in GER between primary and secondary education which supports this analysis. We also found that the difference between the female and male enrollment has rapidly decreased within the past 8 years. 
