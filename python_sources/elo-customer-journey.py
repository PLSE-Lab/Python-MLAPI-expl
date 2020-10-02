#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#New to Kaggle and this is my first public kernel. Hope you find it interesting or helpful! This kernel provides visualizations on an individual card's journey. You can fork this to explore the training set in search of patterns.

#The final cell provides the only output, a plot of every transaction made by a specified card. The columns on the left are subsector_id, merchant_category_id, and merchant_id, respectively. Each point is a transaction and if you hover over one in an interactive notebook (doesn't work in the kernel output, you'll have to fork to see) you can see the transaction details.

#To use the forked notebook, edit the last cell. You can call the function on any card or row number in the train set.

# Change the below to true before running this kernel. Then run the kernel end-to-end. Then you can pick cards to explore in the last cell, just rerunning the last cell
interactive = False


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

if interactive:
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 60

import matplotlib.pyplot as plt
from matplotlib import patches

import datetime
from dateutil.relativedelta import relativedelta

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merchants = pd.read_csv('../input/merchants.csv')
trans = pd.read_csv('../input/historical_transactions.csv')
new_trans = pd.read_csv('../input/new_merchant_transactions.csv')


# In[ ]:


def get_journey_data(cx_id):
    # select customer
    if type(cx_id) is int:
        cx = train.iloc[cx_id]
        cx_id = cx['card_id']
    else:
        cx = train[train['card_id'] == cx_id].iloc[0]
        
    # select transactions
    cx_hist_trans = trans[trans['card_id']==cx_id]
    cx_new_trans = new_trans[new_trans['card_id']==cx_id]

    # designate transaction source and merge transaction lists
    cx_hist_trans = cx_hist_trans.assign(Source='Historical')
    cx_new_trans = cx_new_trans.assign(Source='New')
    cx_trans = pd.concat((cx_hist_trans, cx_new_trans), axis=0).reset_index()

    # decompose purchase date
    trans_date = pd.DataFrame(index=cx_trans.index)
    trans_date['Timestamp'] = pd.to_datetime(cx_trans['purchase_date'], format='%Y-%m-%d %H:%M:%S')
    dates, times, years, months, days = zip(*[(d.date(), d.time(), d.year, d.month, d.day) for d in trans_date['Timestamp']])
    trans_date = trans_date.assign(Date=dates, Time=times, Year=years, Month=months, Day=days)
    cx_trans = pd.concat((cx_trans, trans_date), axis=1)
    
    return cx, clean_trans(cx_trans)

def clean_trans(trans):
    trans['merchant_id'] = trans['merchant_id'].where(~trans['merchant_id'].isna(), 'Missing')
    trans['merchant_category_id'] = trans['merchant_category_id'].where(~trans['merchant_category_id'].isna(), 'Missing')
    trans['subsector_id'] = trans['subsector_id'].where(~trans['subsector_id'].isna(), 'Missing')
    return trans


# In[ ]:


def print_head(cx, cx_trans):
    print('{:30}'.format('Card:'), cx['card_id'])
    print('{:30}'.format('Active Date:'), cx['first_active_month'])
    print('{:30}'.format('Target:'), cx['target'])
    print('')
    print('{:30}'.format('# of Transactions:'), len(cx_trans))
    print('{:30}'.format('# of New Merchants:'), sum(cx_trans['Source']=='New'))
    print('{:30}'.format('% Authorized:'), str(int(10000*sum(cx_trans['authorized_flag']=='Y')/len(cx_trans))/100.) + '%')
    print('{:30}'.format('Total Spend:'), sum(cx_trans['purchase_amount']))


# In[ ]:


#outstanding:
# 1. hover annotate
# 2. spread/swarm nearby points

merchant_width = 40
group_width = 15
key_date_offset = 0.8

interactive = False

points_with_annotation = []

def plot_journey(cx, trans):    
    date_range = get_date_range(trans)
    merchant_info = analyze_merchants(trans)
    
    fig, axes = create_figure(merchant_info, date_range, x_offset=-(merchant_width+2*group_width), y_offset=-key_date_offset)
    
    draw_swimlanes(merchant_info, title_size=-merchant_width)
    draw_groups(axes, merchant_info, title_size=group_width, x_offset=-merchant_width)
    
    draw_start_line()
    draw_first_active_month(cx, date_range['min'], -key_date_offset)
    draw_lag_ref_month(trans, date_range['min'], -key_date_offset)
    
    draw_transactions(axes, trans, merchant_info, date_range['min'])
    
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    
def draw_lag_ref_month(trans, ref_date, y_offset, pad=0.3, c="#666666", a=0.75):
    dates = [(x['Date'] + relativedelta(months=-x['month_lag'])).replace(day=1) for _, x in trans.iterrows()]
    date = last_day_of_month(max(set(dates), key=dates.count))
    
    x = (date-ref_date).days + 1
    plt.annotate('Lag Reference', (x, y_offset+pad), verticalalignment='top', horizontalalignment='center')
    plt.axvline(x, color=c, alpha=a)
    
def draw_first_active_month(cx, ref_date, y_offset, pad=0.3, c="#eeeeee", a=0.75):
    active_date = datetime.datetime.strptime(cx['first_active_month'], '%Y-%m')
    active_date_end = last_day_of_month(active_date)
    active_start = (active_date.date()-ref_date).days
    active_end = (active_date_end.date()-ref_date).days
    
    if active_end > 0:
        if active_start < 0:
            active_start = 0
        plt.annotate('First Active Month', ((active_start+active_end)/2, y_offset+pad), verticalalignment='top', horizontalalignment='center')
        plt.axvspan(active_start, active_end, color=c, alpha=a)
    
def draw_start_line(c="#eeeeee", a=0.75):
    plt.axvline(0, color=c, alpha=a)

def draw_transactions(ax, trans, info, ref_date, hist_c='b', new_c='r', size=8, tooltip_offset=0.1):
    for _, t in trans.iterrows():
        x = (t['Date'] - ref_date).days
        
        y = info['merchant_pos'][stringify_merchant(t['subsector_id'], t['merchant_category_id'], t['merchant_id'])]
        
        c = hist_c
        if t['Source'] == 'New':
            c = new_c
            
        point, = ax.plot(x, y, 'o', markersize=size, color=c)
        
        y_step = 0
        if y < 3:
            y_step = 3
        annotation = ax.annotate(get_annotation(t),
        xy=(x, y), xycoords='data',
        xytext=(x + tooltip_offset, y + tooltip_offset + y_step), textcoords='data',
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.5", alpha=0.9)
        )
        annotation.set_visible(False)
        points_with_annotation.append([point, annotation])

def draw_swimlanes(info, title_size, pad=3, c='#eeefff'):
    i = 0
    for key in info['merchant_pos']:
        i = i+1
        pos = info['merchant_pos'][key]
        size = info['merchant_size'][key]
        title, _, _ = de_stringify_merchant(key)
        
        if i % 2 == 0:
            plt.axhspan(pos-size/2, pos+size/2, color=c)
        plt.annotate(title, (title_size+pad, pos), verticalalignment='center')

def create_figure(info, date_range, x_offset, y_offset):
    if interactive:
        get_ipython().run_line_magic('matplotlib', 'notebook')
    f = plt.figure()
    ax = plt.axes()
    
    # dimensions
    height = sum(info['merchant_size'].values())
    width = (date_range['max'] - date_range['min']).days
    
    # chart size
    f.set_size_inches(23, 0.5*(height - y_offset))
    f.tight_layout(pad=2)
    
    # y axis
    ax.set_ylim(y_offset, height)
    ax.invert_yaxis()
    ax.set_yticks([])
    
    # x axis
    ax.xaxis.tick_top()
    ax.set_xlim(x_offset, width)
    x_ticks = get_ticks(date_range)
    ax.set_xticks(x_ticks['Ticks'])
    ax.set_xticklabels(x_ticks['Labels'])
    
    return f, ax

def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) - datetime.timedelta(days=1)

def get_date_range(trans):
    return {'min': min(trans['Date']), 'max': max(trans['Date'])}

def get_ticks(date_range):
    dates = [date_range['min'].replace(day=15)]
    while dates[-1] < date_range['max']:
        dates.append((dates[-1] + relativedelta(months=1)).replace(day=15))
    if date_range['min'].day > 15:
        del dates[0]
        
    ticks = [(d - date_range['min']).days for d in dates]
    labels = [d.strftime('%b') + ' ' + str(d.year) for d in dates]
    return pd.DataFrame({'Ticks': ticks, 'Labels': labels})
            
def analyze_merchants(trans):
    groups = {}
    merchant_pos = {}
    merchant_size = {}
    
    subsectors = get_sorted_merchant_group(trans, 'subsector_id')
    for subsector in subsectors:
        groups[subsector] = {}
        sector_trans = trans[trans['subsector_id']==subsector]
        merchant_categories = get_sorted_merchant_group(sector_trans, 'merchant_category_id')
        
        for merchant_category in merchant_categories:
            groups[subsector][merchant_category] = []
            category_trans = sector_trans[sector_trans['merchant_category_id']==merchant_category]
            merchants = get_sorted_merchant_group(category_trans, 'merchant_id')
            
            for merchant in merchants:
                groups[subsector][merchant_category].append(merchant)
                merchant_trans = category_trans[category_trans['merchant_id']==merchant]
                
                merch_str = stringify_merchant(subsector, merchant_category, merchant)
                merchant_size[merch_str] = calc_merchant_size(merchant_trans)
                merchant_pos[merch_str] = sum(merchant_size.values()) - merchant_size[merch_str]/2.
    
    return {'groups': groups, 'merchant_pos': merchant_pos, 'merchant_size': merchant_size}
    
def calc_merchant_size(trans):
    return 1.

def stringify_merchant(subsector, category, merchant):
    return str(subsector) + '.' + str(category) + '.' + str(merchant)

def de_stringify_merchant(merchant_string):
    subsector, category, merchant = merchant_string.split('.')
    return merchant, category, subsector
            
def get_sorted_merchant_group(cx_trans, group):
    return list(cx_trans[group].value_counts().index)

def draw_groups(ax, info, title_size, x_offset, x_pad=1, y_pad=0.1, sub_fill='#aaafff', cat_fill='#cccfff'):
    plt.axvspan(x_offset-2*title_size, x_offset-x_pad, color='#ffffff')
    
    keys = np.array([de_stringify_merchant(x) for x in info['merchant_pos'].keys()])
    subsectors = np.unique(keys[:, 2])
    for subsector in subsectors:
        sub_keys = (keys[:, 2] == subsector)
        sub_pos = np.array(list(info['merchant_pos'].values()))[sub_keys]
        sub_size = np.array(list(info['merchant_size'].values()))[sub_keys]
        
        x = x_offset - 2 * title_size + x_pad
        y = sub_pos[0] - sub_size[0]/2 + y_pad
        width = title_size - x_pad
        height = sum(sub_size) - y_pad
        
        ax.add_patch(patches.Rectangle((x ,y), width, height, color=sub_fill))
        plt.annotate(subsector, (x+width/2, y+height/2), verticalalignment='center', horizontalalignment='center')
        
        categories = np.unique(keys[sub_keys, 1])
        for category in categories:
            cat_keys = (keys[:, 1] == category) & sub_keys
            cat_pos = np.array(list(info['merchant_pos'].values()))[cat_keys]
            cat_size = np.array(list(info['merchant_size'].values()))[cat_keys]     
        
            x = x_offset - title_size + x_pad
            y = cat_pos[0] - cat_size[0]/2 + y_pad
            width = title_size - x_pad
            height = sum(cat_size) - y_pad

            ax.add_patch(patches.Rectangle((x ,y), width, height, color=cat_fill))
            plt.annotate(category, (x+width/2, y+height/2), verticalalignment='center', horizontalalignment='center')    
            
def on_move(event):
    visibility_changed = False
    for point, annotation in points_with_annotation:
        should_be_visible = (point.contains(event)[0] == True)

        if should_be_visible != annotation.get_visible():
            visibility_changed = True
            annotation.set_visible(should_be_visible)

    if visibility_changed:        
        plt.draw()
        
def get_annotation(transaction):
    string = ''
    cats = {
        'purchase_date': 'Date',
        'purchase_amount': 'Purchase Amount',
        'installments': 'Installments',
        'authorized_flag': 'Approved',
        'city_id': 'City',
        'state_id': 'State',
        'category_1': 'Category 1',
        'category_2': 'Category 2',
        'category_3': 'Category 3',
    }
    for c in cats:
        string += c + ': ' + str(transaction[c]) + '\n'
    
    return string[:-1]


# In[ ]:


def get_journey(customer_id):
    cx, cx_trans = get_journey_data(customer_id)
    print_head(cx, cx_trans)
    plot_journey(cx, cx_trans)  


# In[ ]:


# this is just a placeholder so I can submit this kernel
df = pd.DataFrame({'card_id': test['card_id'], 'target': np.zeros(test['card_id'].shape)})
df.to_csv('submission.csv', index=False)


# In[ ]:


# Enter the card_id you would like to explore or an integer to select cards by row
#get_journey(2) EXAMPLE
#get_journey('C_ID_d639edf6cd') EXAMPLE

# you can edit this
get_journey(1)

