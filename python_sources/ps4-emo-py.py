# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


def date_fixer(datestring):
    '''
    Takes a datestring in the format MM/DD/YYYY and converts it to a datetime
    object.

    Input:  (string) datestring: a date in MM/DD/YY format

    Output: (datetime) date: the date as a datetime object
    '''

    date = dt.datetime.strptime(datestring,'%m/%d/%Y')

    return(date)
    
    

if __name__ == "__main__":
# Any results you write to the current directory are saved as output.



    f = '../input/BPD_Arrests.csv' #../input/BPD_Arrests.csv
    names = ['Arrest',
             'Age',
             'Sex',
             'Race',
             'ArrestDate',
             'ArrestTime',
             'ArrestLocation',
             'IncidentOffense',
             'IncidentLocation',
             'Charge',
             'ChargeDescription',
             'District',
             'Post',
             'Neighborhood',
             'Location 1']

    max_date = date_fixer('01/01/2016')

    # Read in dataset as pandas dataframs
    df = pd.read_csv(f, converters={'ArrestDate': date_fixer})
    d1 = df[df.ArrestDate < max_date]
    d1['Month'] = df.ArrestDate.map(lambda x: x.month)

    
    #arrests = d1.groupby('Month').count().Arrest.tolist()
    months = d1.Month
    obs = len(months)
    wghts = (1 / float(obs)) * np.ones(obs)
    
    
    fig, ax = plt.subplots()



    #total_arrests = sum(arrests)
    c = 'cyan'
    lbls = ['January',
            'February',
            'March',
            'April',
            'May',
            'June',
            'July',
            'August',
            'September',
            'October',
            'November',
            'December']
    
    bins = np.arange(1,14,1)
    ticks = np.arange(1.5,13,1)
    start_bin = 1
    stop_bin = 14#12.5#14

    # Plot the histogram, set the ticks and labels, and save the figure
    plt.hist(months, bins = bins, weights = wghts, color=c)
    ax.xaxis.set_ticks(ticks, minor=False)#[.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5],minor=False)
    #ax.xaxis.set_ticks(np.arange(start_bin + .5, stop_bin - .5, .95))
    ax.set_xticklabels(lbls, fontsize=7)
    plt.title('Baltimore Arrests 2013-2015, Grouped by Month,\nas a Proportion of Total Arrests', fontsize=14)
    plt.xlabel('Arrest Month')
    plt.ylabel('Proportion of Total Arrests')
    plt.xlim((1,13))
    #output_path = os.path.join(output_dir, 'Fig_2')
    #plt.savefig(output_path)
    plt.savefig('Fig')
    #plt.show()
    plt.close()