#Test wheather there are a different number of Reddit comments posted on weekdays than on weekends?



import sys

import numpy as np

import pandas as pd

import gzip

from scipy import stats

import time

from datetime import date

from scipy.stats import mannwhitneyu







OUTPUT_TEMPLATE = (

    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"

    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"

    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"

    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"

    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"

    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"

    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"

    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"

    "Mann–Whitney U-test p-value: {utest_p:.3g}"

)





def main():



    #Load data

    rd = pd.read_json('../input/reddit-counts.json', lines=True)

    

    #Filter data

    rd = rd.loc[rd['subreddit'] == 'canada']

    rd = rd[rd['date'].isin(pd.date_range("2012-01-01", "2013-12-31"))]

    rd['weekday'] = rd['date'].apply(lambda x:x.weekday())

    wkday = rd.loc[rd['weekday'] < 5]

    wkend = rd.loc[rd['weekday'] >=5]



    ## Student's T-Test

    init_ttest = stats.ttest_ind(wkday['comment_count'], wkend['comment_count']) #pvalue=1.30055e-58 >0.05

    init_wkday_normality = stats.normaltest(wkday['comment_count']) #pvalue=1.009e-07 <0.05, not normal distribution

    init_wkend_normality = stats.normaltest(wkend['comment_count']) #pvalue=0.001520 <0.05, not normal distribution

    init_levene = stats.levene(wkday['comment_count'],wkend['comment_count']) #pvalue=0.043787 <0.05, not equal variances

    

    ## Transforming data

    transf_wkday = np.sqrt(wkday['comment_count'])

    transformed_wkday_nomality = stats.normaltest(transf_wkday) #pvalue = 0.036872 < 0.05, not normal distribution

    

    transf_wkend = np.sqrt(wkend['comment_count'])

    transformed_wkend_nomality = stats.normaltest(transf_wkend) #pvalue = 0.107605 > 0.05, normal distribution



    transf_levene = stats.levene(transf_wkday,transf_wkend) #pvalue=0.55605 <0.05, equal variances



    ## Central Limit Theorem

    # Obtain count mean from weekday

    wkday = rd.loc[rd['weekday'] < 5].reset_index() #in case of SettingWithCopyWarning

    #isocanlendar covertion: get a “year” and “week number” from the first two values returned

    wkday['year'] = wkday['date'].apply(lambda x: date.isocalendar(x)[0]) 

    wkday['week'] = wkday['date'].apply(lambda x: date.isocalendar(x)[1]) 

    wkday_m = wkday.groupby(['year','week'])['comment_count'].mean()



    # Obtain count mean from weekend

    wkend = rd.loc[rd['weekday'] >=5].reset_index()

    #isocanlendar covertion: get a “year” and “week number” from the first two values returned

    wkend['year'] = wkend['date'].apply(lambda x: date.isocalendar(x)[0])

    wkend['week'] = wkend['date'].apply(lambda x: date.isocalendar(x)[1]) 

    wkend_m = wkend.groupby(['year','week'])['comment_count'].mean()



    wkday_normality = stats.normaltest(wkday_m) #pvalue=0.308263 >0.05, so it's normal distribution

    wkend_normality = stats.normaltest(wkend_m) #pvalue=0.152949 >0.05, so it's normal distribution

    wk_levene = stats.levene(wkday_m, wkend_m)  #pvalue=0.203838 >0.05, so => equal variance



    ttest = stats.ttest_ind(wkday_m, wkend_m) #pvalue=1.33536 >0.05



    ## Non-parametric test

    utest = mannwhitneyu(wkday['comment_count'], wkend['comment_count']) #pvalue = 4.31222 





    print(OUTPUT_TEMPLATE.format(

            initial_ttest_p= init_ttest.pvalue,

	    initial_weekday_normality_p= init_wkday_normality.pvalue,

	    initial_weekend_normality_p= init_wkend_normality.pvalue,

	    initial_levene_p= init_levene.pvalue,

	    transformed_weekday_normality_p= transformed_wkday_nomality.pvalue,

	    transformed_weekend_normality_p= transformed_wkend_nomality.pvalue,

	    transformed_levene_p= transf_levene.pvalue,

	    weekly_weekday_normality_p= wkday_normality.pvalue,

	    weekly_weekend_normality_p= wkend_normality.pvalue,

	    weekly_levene_p= wk_levene.pvalue,

	    weekly_ttest_p= ttest.pvalue,

	    utest_p= utest.pvalue,

    ))





if __name__ == '__main__':

    main()
