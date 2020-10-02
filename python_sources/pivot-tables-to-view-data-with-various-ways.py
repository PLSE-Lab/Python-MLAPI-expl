# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
print(df.columns)

df_jobtype_salary_promotion = pd.pivot_table(df,
                        values = ['satisfaction_level', 'last_evaluation'],
                        index = ['sales', 'salary', 'promotion_last_5years'],
                        columns = [],aggfunc=[np.mean], 
                        margins=True).fillna('')
                        
df_jobtype_salary_promotion
                        
df_jobtype_salary_time_hours = pd.pivot_table(df,
                        values = ['satisfaction_level', 'last_evaluation'],
                        index = ['sales', 'salary', 'time_spend_company', 'average_montly_hours'],
                        columns = [],aggfunc=[np.mean], 
                        margins=True).fillna('')
df_jobtype_salary_time_hours
