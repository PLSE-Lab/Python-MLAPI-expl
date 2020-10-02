#!/usr/bin/env python
# coding: utf-8

# # IoT Data Processing

# Here we show how to extract relevant info from the pcap and convert them in csv format. 

# ## IoT DATA

# In this lab we use data from network traffic from IoT devices located in US and UK.
# * **pcaps from Controlled experiment traffic**: These consist of actively interacting with IoT devices and then labeling the captured traffic with the interaction name. For each of these experiments we first wait for the device to be powered on for at least two minutes (to avoid including power experiments traffic). After two minutes, and right before the interaction starts, we begin capturing the traffic and continue to do so for the entire duration of the interaction (i.e., switching on the smart bulb through the app)
# 
# * **CSV from Controlled experiment 1 min sample**: We extracted relevant info from the pcap and convert them in csv format for further processing on 1 minute sample.
# 
# *Format*: 
# timestamp (unix epoch), bandwidth up(kbits/s), bandwidth down(kbits/s), connections made, different ports accessed, different IPs accessed
# 
# * **CSV from Controlled experiment destination analysis**: We extracted relevant info from the pcap and convert them in csv format for further processing about destinations contacted. 
# 
# *Format*: index,number,device,ip,host,host_full,traffic_snd,traffic_rcv,packet_snd,packet_rcv,country,party,lab,experiment,network,input_file,organisation,category
# 

# ## From pcap to CSV

# tshark -r data/pcap/uk/blink-security-hub/power/power1.pcap -T fields -e frame.number -e frame.time_epoch -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -e tcp.srcport -e udp.srcport -e tcp.dstport -e udp.dstport -e frame.len -E header=y -E separator=, -E quote=n -E occurrence=f
# 

# ### Import Required Packages

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd
from datetime import datetime
import sys

#!pip install matplotlib


# #### Reading data 1 min sample

# Let's first read the data per device and country

# In[ ]:


blink_security_hub_uk = pd.read_csv('../input/csv/one_min_sample/uk/uk_tot_blink-security-hub.csv')
blink_security_hub_uk


# In[ ]:


blink_security_hub_uk = pd.read_csv('../input/csv/one_min_sample/uk/uk_tot_blink-security-hub.csv')

echoplus_uk = pd.read_csv('../input/csv/one_min_sample/uk/uk_tot_echoplus.csv')

magichome_strip_uk = pd.read_csv('../input/csv/one_min_sample/uk/uk_tot_magichome-strip.csv')

samsungtv_uk = pd.read_csv('../input/csv/one_min_sample/uk/uk_tot_samsungtv-wired.csv')

philips_hub_uk = pd.read_csv('../input/csv/one_min_sample/uk/uk_tot_t-philips-hub.csv')

blink_security_hub_us = pd.read_csv('../input/csv/one_min_sample/us/us_tot_blink-security-hub.csv')

echoplus_us = pd.read_csv('../input/csv/one_min_sample/us/us_tot_echoplus.csv')

magichome_strip_us = pd.read_csv('../input/csv/one_min_sample/us/us_tot_magichome-strip.csv')

samsungtv_us = pd.read_csv('../input/csv/one_min_sample/us/us_tot_samsungtv-wired.csv')

philips_hub_us = pd.read_csv('../input/csv/one_min_sample/us/us_tot_t-philips-hub.csv')

philips_hub_us


# ### Destination Analysis

# #### Reading data per categories for destination analysis

# Let's read the data per destination analysis 

# In[ ]:


all_party = pd.read_csv('../input/csv/parties/all_party.csv')

all_party


# ### Data Visualisation 

# #### Count parties

# Let's count and list the number of destinations contacted over time for patterns recognition

# In[ ]:


all_data = all_party

all_data['uk'] = '0'
all_data['us'] = '0'
all_data.loc[(all_data['lab'] == 'icl'), 'uk'] = '1'
all_data.loc[(all_data['lab'] == 'neu'), 'us'] = '1'
#excluding unknown
all_data.loc[all_data['host'].str.contains('samsungcloud.com'), 'party'] = 'first party'
all_data.loc[all_data['host'].str.contains('amazon'), 'party'] = 'support party'
all_data.loc[all_data['host'].str.contains('tplinkra.com'), 'party'] = 'first party'
all_data.loc[all_data['host'].str.contains('fastly'), 'party'] = 'support party'


all_data_un=all_data.loc[~(all_data['party'] == 'first party')&~(all_data['lab'] == 'neu')&~(all_data['party'] == 'support party')&~(all_data['experiment'] == 'google')]

device_only_us_uk = ['cloudcam', 'ikettle', 'bulb1', 'xiaomi-ricecooker', 'brewer', 'washer','microseven-camera',
                     'invoke', 'dlink-mov', 'microwave', 'wink-hub2', 'dryer', 'lgtv-wired', 'amcrest-cam-wired', 'lefun-cam-wired', 'xiaomi-strip', 'fridge', 'zmodo-doorbell','luohe-spycam','xiaomi-cleaner' , 'netatmo-weather-station', 'google-home', 'bosiwo-camera-wifi', 'honeywell-thermostat', 'iphone', 'charger-camera', 'xiaomi-cam2', 'allure-speaker', 'smarter-coffee-mach', 'bosiwo-camera-wired','xiaomi-plug','tplink-plug2','t-smartthings-hub', 'galaxytab-a','google-home-mini2', 'philips-bulb']

data = all_data_un[~all_data_un['device'].isin(device_only_us_uk)]

third_pary_list=data["host"].unique()
print (len(third_pary_list))
print (third_pary_list)

third_pary_list=all_data_un["host"].unique()
print (len(third_pary_list))
print (third_pary_list)


# #### Plot destinations UK, US

# Let's count the number of parties contacted over time for different regions

# In[ ]:


import matplotlib.pyplot as plt

all_data['uk'] = '0'
all_data['us'] = '0'
all_data.loc[(all_data['lab'] == 'icl'), 'uk'] = '1'
all_data.loc[(all_data['lab'] == 'neu'), 'us'] = '1'
#excluding unknown
all_data_un=all_data.loc[~(all_data['party'] == '0')]

all_clean_dup = all_data_un.drop_duplicates(subset=['device','lab','host','party'], keep='first')

label_count = all_clean_dup.groupby(['device','lab','party']).size().to_frame('count').reset_index()

label_count
df=label_count

df.loc[(df['lab'] == 'icl') & (df['party'] == 'first party'), 'label'] = 'UK first party'
df.loc[(df['lab'] == 'icl') & (df['party'] == 'support party'), 'label'] = 'UK support party'
df.loc[(df['lab'] == 'icl') & (df['party'] == 'third party'), 'label'] = 'UK third party'

df.loc[(df['lab'] == 'neu') & (df['party'] == 'first party'), 'label'] = 'US first party'
df.loc[(df['lab'] == 'neu') & (df['party'] == 'support party'), 'label'] = 'US support party'
df.loc[(df['lab'] == 'neu') & (df['party'] == 'third party'), 'label'] = 'US third party'
print(df)

colors = ["#c3f7c3", "#95f995","#049304","#87CEFA", "#1E90FF","#0000CD"]

fig, ax = plt.subplots(figsize=(17,7))

pivot_df = df.pivot(index='device', columns='label', values='count')
pivot_df.plot(ax=ax, kind='bar', stacked=True, color=colors)  # pass in subplot as an argument


plt.tight_layout()
plt.show()


# #### Plot destinations UK, US

# Let's count the number of parties contacted over time for different devices

# In[ ]:


all_data_un = all_party

all_clean_dup = all_data_un.drop_duplicates(subset=['device','host','party'], keep='first')


#count party and plot with 0
label_count = all_clean_dup.groupby(['device','party']).size().to_frame('count').reset_index()
df=label_count

fig, ax = plt.subplots(figsize=(17,7))
device_list = df["device"].unique()

s = label_count.reset_index().pivot(index='device', columns='party' , values='count').plot(ax=ax, kind='bar')
plt.tight_layout()
plt.show()

#count party and plot without 0

label_count_nozero = label_count[label_count.party!='0']

fig, ax = plt.subplots(figsize=(15,7))

device_list = df["device"].unique()

s = label_count_nozero.reset_index().pivot(index='device', columns='party' , values='count').plot(ax=ax, kind='bar')
ax.set_ylabel("# of unique destinations", fontsize=15)
ax.set_xlabel("device", fontsize=15)
plt.tight_layout()
plt.show()


# #### Plot destinations UK, US

# Let's count the number of parties contacted over time for different categories

# In[ ]:


all_data_un = all_party

all_clean_dup = all_data_un.drop_duplicates(subset=['category','host','party'], keep='first')


#count party and plot with 0
label_count = all_clean_dup.groupby(['category','party']).size().to_frame('count').reset_index()
df=label_count

fig, ax = plt.subplots(figsize=(17,7))
category_list = df["category"].unique()

s = label_count.reset_index().pivot(index='category', columns='party' , values='count').plot(ax=ax, kind='bar')
plt.tight_layout()

#count party and plot without 0

label_count_nozero = label_count[label_count.party!='0']

fig, ax = plt.subplots(figsize=(15,7))

category_list = df["category"].unique()

s = label_count_nozero.reset_index().pivot(index='category', columns='party' , values='count').plot(ax=ax, kind='bar')
plt.tight_layout()
ax.set_ylabel("# of unique destinations", fontsize=15)
ax.set_xlabel("category", fontsize=15)
plt.show()


# #### Comparing Destinations UK, US

# Let's count the number of destinations in different regions

# In[ ]:


import glob
import os
import itertools
from pandas import Series

marker = itertools.cycle((',', '+', '.', 'o', '*'))

#read uk
data=pd.read_csv('../input/csv/experiments/uk_tagged_non-vpn_new')

idle=pd.read_csv('../input/csv/experiments/uk_idle_non-vpn_new')

all_uk = idle.append(data)

#read us
data_us=pd.read_csv('../input/csv/experiments/us_tagged_non-vpn_new')

idle_us=pd.read_csv('../input/csv/experiments/us_idle_non-vpn_new')

all_us = idle.append(data_us)

all = all_uk.append(all_us)

device_list = all["device"].unique()

pattern = '|'.join(device_list)

all_clean = all[(~all['host'].str.contains(":")) & (~all['host'].str.contains("192.168.*")) & (~all['host'].str.contains("amazonaws")) & (~all['host'].str.contains("ic.ac.uk"))& (~all['host'].str.contains("255.255.255.255"))& (~all['host'].str.contains("224.0.0.*"))& (~all['host'].str.contains("224.0.1.*"))& (~all['host'].str.contains("neu.edu"))& (~all['host'].str.contains("239.255.*"))& (~all['host'].str.contains("mitm"))& (~all['host'].str.contains(pattern))& (~all['host'].str.contains("nexus"))& (~all['host'].str.contains("galaxy"))& (~all['device'].str.contains("iphone"))& (~all['device'].str.contains("nexus"))& (~all['host'].str.contains("ntp.org"))& (~all['host'].str.startswith("2012"))]

all_clean.loc[all_clean['host'].str.startswith('cloudfront'), 'host'] = 'amazon'
all_clean.loc[all_clean['host'].str.contains('cloudapp'), 'host'] = 'microsoft'
all_clean.loc[all_clean['host'].str.contains('azure.com'), 'host'] = 'microsoft'
all_clean.loc[all_clean['host'].str.contains('gvt2.com'), 'host'] = 'google'
all_clean.loc[all_clean['host'].str.contains('gstatic.com'), 'host'] = 'google'
all_clean.loc[all_clean['host'].str.contains('google'), 'host'] = 'google'
all_clean.loc[all_clean['host'].str.startswith('nflxso'), 'host'] = 'netflix.com'
all_clean.loc[all_clean['host'].str.startswith('1e100'), 'host'] = 'google'
all_clean.loc[all_clean['host'].str.contains('12.167.151.2'), 'host'] = 'att.com'
all_clean.loc[all_clean['host'].str.contains('microsoft'), 'host'] = 'microsoft'
all_clean.loc[all_clean['host'].str.contains('104.171.118.254'), 'host'] = 'bigbrainglobal.com'
all_clean.loc[all_clean['host'].str.startswith('216.151.187.'), 'host'] = 'BandCon'
all_clean.loc[all_clean['host'].str.startswith('amaz.'), 'host'] = 'amazon'
all_clean.loc[all_clean['host'].str.startswith('47.'), 'host'] = 'alibaba'
all_clean.loc[all_clean['host'].str.contains('alibaba'), 'host'] = 'alibaba'
all_clean.loc[all_clean['host'].str.contains('amazon'), 'host'] = 'amazon'
all_clean.loc[all_clean['host'].str.contains('akamai'), 'host'] = 'akamai'
print("dest #")
print (len(all_clean["host"].unique()))

host_ex_count = all_clean.groupby(['device','input_file','host','network'])

all_clean_dup = all_clean.drop_duplicates(subset=['device','input_file','host','network'], keep='first')

host_ex_count = all_clean_dup.groupby(['device','experiment','host','network']).size().to_frame('size').reset_index()

exp_list = host_ex_count["experiment"].unique()
for exp in exp_list:
    host_ex_count.loc[(host_ex_count['network'] == 'icl') & (host_ex_count['experiment'] == exp), 'label'] = 'icl_'+str(exp)
    host_ex_count.loc[(host_ex_count['network'] == 'neu') & (host_ex_count['experiment'] == exp), 'label'] = 'neu_'+str(exp)
host_ex_count_icl=host_ex_count.loc[(host_ex_count['network'] == 'icl')]
host_ex_count_neu=host_ex_count.loc[(host_ex_count['network'] == 'neu')]

host_list = host_ex_count["host"].unique()

fig, ax = plt.subplots(figsize=(15,7))
host_ex_count2 = host_ex_count.groupby(['device','host']).size().to_frame('size').reset_index()
host_ex_count3 = Series(host_ex_count2['device'])

print (host_ex_count3)
vc = host_ex_count3.value_counts()
print(vc)
fig, ax = plt.subplots(figsize=(15,7))
plt.s=vc.plot.bar(ax=ax)
ax.set_ylabel("# of unique destinations", fontsize=12)
ax.set_xlabel("Device name", fontsize=12)
plt.tight_layout()
fig.show()
#fig.savefig('../fig/total_uk_us_non-vpn_all.eps')


print(host_ex_count.to_csv('host_ex_count.csv'))
pivot_tab = host_ex_count.pivot_table(
                                      values='size',
                                      index=['device', 'host'],
                                      columns='network',
                                      aggfunc=np.sum)

#print per device
host_count_lab_device = all_clean_dup.groupby(['device','host','network']).size().to_frame('size').reset_index()
pivot_tab2 = host_count_lab_device.pivot_table(values='host', index='device', columns='network',
                                               aggfunc=lambda x: len(x.unique()))

#print per device per experiment
host_count_lab_exp = all_clean_dup.groupby(['device','host','experiment','network']).size().to_frame('size').reset_index()
pivot_tab_exp = host_count_lab_exp.pivot_table(values='host', index=['device','experiment'], columns='network',
                                               aggfunc=lambda x: len(x.unique()))

pivot_tab.loc[(pivot_tab['icl']>=1) & (pivot_tab['neu']>1), 'common'] = 'both'
pivot_tab.loc[(pivot_tab['icl']>=1) & (pivot_tab['neu'].isnull()), 'common'] = 'icl'
pivot_tab.loc[(pivot_tab['neu']>=1) & (pivot_tab['icl'].isnull()), 'common'] = 'neu'
print("dest icl")
print(pivot_tab.loc[(pivot_tab['icl']>=1) & (pivot_tab['neu'].isnull())].to_csv('only_icl.csv'))
print("dest neu")
print(pivot_tab.loc[(pivot_tab['neu']>=1) & (pivot_tab['icl'].isnull())].to_csv('only_neu.csv'))

flattened = pd.DataFrame(pivot_tab.to_records())

#plot number of unique destination
fig, ax = plt.subplots(figsize=(15,7))
plt.s=flattened['common'].value_counts().plot.bar(ax=ax)
ax.set_ylabel("# of unique destinations", fontsize=20)
plt.tight_layout()
fig.show()
#fig.savefig('../fig/total_uk_us_non-vpn.eps')

fig, ax = plt.subplots(figsize=(15,7))
plt.s=flattened.groupby(['device', 'common']).size().unstack().plot(ax=ax, kind='bar', stacked=True)
ax.set_ylabel("# of unique destinations", fontsize=20)
plt.tight_layout()
fig.show()
#fig.savefig('../fig/total_uk_us_non-vpn_device.eps')

device_only_us = ['cloudcam', 'ikettle', 'bulb1', 'xiaomi-ricecooker', 'brewer', 'washer','microseven-camera',
                  'invoke', 'dlink-mov', 'microwave', 'wink-hub2', 'dryer', 'lgtv-wired', 'amcrest-cam-wired', 'lefun-cam-wired', 'xiaomi-strip', 'fridge', 'zmodo-doorbell','luohe-spycam']
device_only_uk = ['xiaomi-cleaner' , 'netatmo-weather-station', 'google-home', 'xiaomi-hub', 'bosiwo-camera-wifi', 'honeywell-thermostat', 'iphone', 'charger-camera', 'xiaomi-cam2', 'allure-speaker', 'smarter-coffee-mach', 'bosiwo-camera-wired','xiaomi-plug','tplink-plug2']

flattened = flattened[~flattened['device'].isin(device_only_us)]
flattened = flattened[~flattened['device'].isin(device_only_uk)]

#plot number of unique destination
fig, ax = plt.subplots(figsize=(15,7))
plt.s=flattened['common'].value_counts().plot.bar(ax=ax)
ax.set_ylabel("# of unique destinations", fontsize=20)
plt.tight_layout()
fig.show()
#fig.savefig('../fig/total_uk_us_non-vpn_common.eps')

fig, ax = plt.subplots(figsize=(15,7))
plt.s=flattened.groupby(['device', 'common']).size().unstack().plot(ax=ax, kind='bar', stacked=True)
ax.set_ylabel("# of unique destinations", fontsize=20)
plt.tight_layout()
fig.show()
#fig.savefig('../fig/total_uk_us_non-vpn_device_common.eps')


# #### Plot destinations per experiment per device

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import itertools

data=pd.read_csv('../input/csv/experiments/experiment_uk_tagged_vpn.csv')
print(data)

marker = itertools.cycle((',', '+', '.', 'o', '*'))

idle=pd.read_csv('../input/csv/experiments/experiment_uk_idle_vpn.csv')

all = idle.append(pd.read_csv('../input/csv/experiments/experiment_uk_tagged_vpn.csv'))
print(all)

all_clean = all[(~all['host'].str.contains(":")) & (~all['host'].str.contains("192.168.*")) & (~all['host'].str.contains("amazonaws")) & (~all['host'].str.contains("ic.ac.uk"))& (~all['host'].str.contains("255.255.255.255"))& (~all['host'].str.contains("224.0.0.*"))& (~all['host'].str.contains("224.0.1.*"))& (~all['host'].str.contains("neu.edu"))]

host_ex_count = all_clean.groupby(['device','input_file','host'])

all_clean_dup = all_clean.drop_duplicates(subset=['device','input_file','host'], keep='first')

host_ex_count = all_clean_dup.groupby(['device','experiment','host']).size().to_frame('size').reset_index()

device_list = all_clean["device"].unique()

for dev in device_list:
    dpd = host_ex_count[host_ex_count.device==dev]
    print(dev)
    p = dpd.groupby(['host','experiment']).sum()['size'].unstack()
    print(p)
    fig, ax = plt.subplots(figsize=(15,7))
    s = dpd.groupby(['host','experiment']).sum()['size'].unstack().plot(ax=ax, title = dev, marker='o', linestyle=' ')
    x=p.index
    ax.xaxis.set_ticks(np.arange(len(x)))
    ax.xaxis.set_ticklabels(x, rotation=80, fontsize=20)
    ax.set_ylabel("# of experiments", fontsize=20)
    ax.set_xlabel("host name", fontsize=20)
    plt.tight_layout()
    plt.show()

