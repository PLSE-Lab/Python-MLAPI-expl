#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/employee-compensation.csv')


# ## Initial exploration of data

# In[ ]:


df.head()


# In[ ]:


df['Salaries'].hist(bins=20)


# In[ ]:


total_salary_description = df['Salaries'].describe()
print(total_salary_description)


# Let's explore how many entries with negative salary is presented.

# In[ ]:


negative_salaries = df[df['Salaries'] < 0]
negative_salaries_description = negative_salaries['Salaries'].describe()
print(negative_salaries_description)


# In[ ]:


negative_salaries_description['mean'] / (total_salary_description['count'] / negative_salaries_description['count'])


# On average, negative values would change mean salary by $1.23. Comparing to mean salary of $63,818.59, that value can be neglected, so all rows will be omitted.
# 
# Also all negative values could be inspected with more attention, e.g. what departments are more prone to such values. For the purposes of this research it will be ignored.

# In[ ]:


corrected_df = df[df['Salaries'] > 0].copy() #copying to avoid warnings later


# # Let's explore top departments by average individual salaries and by department salaries budgets.

# In[ ]:


groupped_by_department = corrected_df.groupby(by='Department')
departments_stats = groupped_by_department['Salaries'].agg(['mean', 'sum', 'count', 'std'])
departments_stats['sum'] = departments_stats['sum'] / 1000000000.0

top_departments_by_department_budget = departments_stats.sort_values(by='sum', ascending=False).iloc[0:10]
top_departments_by_average_salary = departments_stats.sort_values(by='mean', ascending=False).iloc[0:10]


# In[ ]:


def plot_salaries(entries, title, field_name):

    def calc_95_ci(entry):
        mean = entry['mean']
        z = 1.960
        std = entry['std']
        n = entry['count']
        diff = z*std/np.sqrt(n)
        return (mean - diff, mean + diff, diff)

    cb_dark_blue = (0/255, 107/255, 164/255)
    cb_orange = (255/255, 128/255, 14/255)
    cb_light_gray = (171/255, 171/255, 171/255)

    fig = plt.figure()
    fig.set_size_inches(18, 12)
    ax1 = fig.add_subplot(111)

    ax1.set_title(title)

    plt.sca(ax1)
    x_labels = entries.index
    plt.xticks(range(10), x_labels, rotation=60)

    sums = entries['sum'].values
    bar1 = ax1.bar([x for x in range(0, 10)], sums, width=0.8, color=cb_dark_blue)
    #ax1.set_ylim([0, 4])
    ax1.set_ylabel("Department " + field_name + " budget, billions of USD")

    ax2 = ax1.twinx()
    means = entries['mean'].values
    bar2 = ax2.bar(range(0, 10), means, width=0.3, color=cb_orange)
    #ax2.set_ylim([0, 170000])
    ax2.set_ylabel("Individual " + field_name + ", USD")

    fig.legend([bar1, bar2], ['Department ' + field_name + ' budget', 'Mean individual ' + field_name], loc=(0.55, 0.9), ncol=2)

    confidence_intervals = [calc_95_ci(entries.loc[x_labels[i]]) for i in range(0, len(x_labels))]
    salaries = entries['mean'].as_matrix()
    for i in range(0, len(x_labels)):
        plt.errorbar(i, salaries[i], xerr=0, yerr=confidence_intervals[i][2], capsize=10, color='black')

    plt.show()


# In[ ]:


plot_salaries(top_departments_by_department_budget, 'Top SF departments by department salary budgets', 'salaries')


# In[ ]:


plot_salaries(top_departments_by_average_salary, 'Top SF departments by average individual salary', 'salaries')


# Now let's explore compensations in details. Salary is only one of components of total compensations.

# # Guaranteed compensations

# Let's explore guaranteed compensations. That includes all benefits and salary, but excludes payments for overtime and other salaries that are not guaranteed in any way.

# In[ ]:


corrected_df['Guaranteed Compensations'] = corrected_df['Salaries'] + corrected_df['Total Benefits']

groupped_by_department = corrected_df.groupby(by='Department')
departments_compensation_stats = groupped_by_department['Guaranteed Compensations'].agg(['mean', 'sum', 'count', 'std'])
departments_compensation_stats['sum'] = departments_compensation_stats['sum'] / 1000000000.0

top_departments_by_compensation_budget = departments_compensation_stats.sort_values(by='sum', ascending=False).iloc[0:10]
top_departments_by_compensation = departments_compensation_stats.sort_values(by='mean', ascending=False).iloc[0:10]


# In[ ]:


plot_salaries(top_departments_by_compensation_budget, 'Top SF departments by department compensation budgets', 'compensations')


# In[ ]:


plot_salaries(top_departments_by_compensation, 'Top SF departments by guaranteed compensation', 'compensations')


# Let's check if top lists for salaries and compensations are identical.

# In[ ]:


all(top_departments_by_compensation_budget.index == top_departments_by_department_budget.index)


# Top lists for department budgets in both categories are identical.

# In[ ]:


all(top_departments_by_compensation.index == top_departments_by_average_salary.index)


# Top lists for individual salaries and compensations are different. Let's explore it.

# In[ ]:


df_by_salary = top_departments_by_average_salary.reset_index(drop=False)[['Department', 'mean']]
df_by_salary[['Department by salary', 'Salary']] = df_by_salary[['Department', 'mean']]
df_by_salary = df_by_salary[['Department by salary', 'Salary']]

df_by_compensation = top_departments_by_compensation.reset_index(drop=False)[['Department', 'mean']]
df_by_compensation[['Department by compensation', 'Compensation']] = df_by_compensation[['Department', 'mean']]
df_by_compensation = df_by_compensation[['Department by compensation', 'Compensation']]

compare_df = pd.concat([df_by_salary, df_by_compensation], axis=1)
compare_df.head(10)


# Top list for individual salaries and compensations are almost identical. The only difference is SHF Sheriff department is slightly better with compensations in comparison with RET Retirement System department.

# While we're here, let's explore how big is the part of benefits for each department in comparison with the salary.

# In[ ]:


direct_comparison_df = top_departments_by_average_salary[['mean']].join(
    top_departments_by_compensation[['mean']], lsuffix=' salary', rsuffix=' compensation')


direct_comparison_df['Benefits to total Compensation, %'] = (
    (direct_comparison_df['mean compensation'] - direct_comparison_df['mean salary']) / direct_comparison_df['mean compensation']) * 100
direct_comparison_df = direct_comparison_df.reset_index(drop=False)
direct_comparison_df.plot(kind='bar',x='Department',y='Benefits to total Compensation, %')


# We can see from the plot that benefits are from ~25 to ~33% of total compensations for the top 10 departments.

# In[ ]:




