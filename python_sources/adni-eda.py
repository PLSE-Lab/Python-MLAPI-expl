# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl # Plotting stuff
import matplotlib.pyplot as plt # Still more plotting stuff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Test the import of the CSV with demographic info
ad_info = pd.read_csv('../input/adni_demographic_master_kaggle.csv')

# Take a look at age distribution
age_hist = ad_info.age_at_scan.hist(bins=15)

age_hist.set_xlabel('Age in years')
age_hist.set_ylabel('Record count')
age_hist.set_title('Age at time of scan',fontsize=16)

plt.savefig('age_hist.png')

# Any results you write to the current directory are saved as output.