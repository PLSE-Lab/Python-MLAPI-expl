#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The goal of this project is to develop a measure of fighter efficiency that can be compared over time and between fighters. Fighter efficiency can be thought of as the percentage of a fighter's attacks that are successful, and these will include attacks of all types (i.e. standing up, on the ground, in the clinch, and takedowns). The problem is, if we only calculate fighter efficiency as the sum of attacks of all types divided by the sum of attempts of all types, we won't be able to easily compare these across fighters. That is because different fighters have different fighting styles. Some of them rely more on takedowns and ground and pound, which tend tend to have relatively higher success rates, while others rely more on standup attacks, which have lower success rates, but higher volume. So, the goal here is to put forward a measure that can be used to standardize for this fact.

# # Getting Started  
# First, let's import the relevant libraries and the raw UFC data.

# In[ ]:


import pandas as pd
raw_data = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv", sep = ";")
raw_data.head()
# Create a copy to work with
df = raw_data.copy()
df.head()


# ### Turning the wide dataframe of fights into a long dataframe of fighters
# We see that the dataframe is in a wide format grouped by each fight, with the Red and Blue corner fighters having their fight stats in columns denoted by 'R_' and 'B_', respectively.  I am going to take the information for red and blue fighters separately, and create a new long dataframe with the stats for every UFC fight for each fighter. In that process, I will also have to work with the raw data to identify the fighter's weightclass, gender, if they won the given fight, and eliminate the precalculated percentage columns, because I want to calculate those more precisely myself. In order to combine all of the fighters into one long dataframe, I will also need to trim the 'R_' and 'B_' prefixes from the columns.

# In[ ]:


# Identify whether or not a fighter won their fight
df['R_win'] = (df['Winner'] == df['R_fighter'])*1
df['B_win'] = (df['Winner'] == df['B_fighter'])*1

# Identifying weight classes
## First, I assign the values of fight_type to a list
vals = list(set(df['Fight_type'].values))

## Second, create a loop over those values that will identify the correct 
## weight class for each fight.
weight_types = []
gender_types = []
title_types = []

for i in vals:
    if 'Women' in i:
        gender_types.append('women')
    else:
        gender_types.append('men')
    if ('Title' in i and 'Tournament' not in i and 'Interim' not in i):
        title_types.append(1)
    else:
        title_types.append(0)
    if 'Catch Weight' in i:
        weight_types.append('Catch weight')
    elif 'Open Weight' in i:
        weight_types.append('Open weight')
    elif 'weight' not in i:
        weight_types.append('Open weight')
    elif 'Light Heavyweight' in i:
        weight_types.append('Light Heavyweight')
    else:
        split_types = i.split(" ")
        for j in split_types:
            if 'weight' in j:
                weight_types.append(j)

## Third, create a dictionary of the original values matched with the 
## weight classes.
weight_dict = dict(zip(vals, weight_types))
gender_dict = dict(zip(vals, gender_types))
title_dict = dict(zip(vals, title_types))

## Fourth, map the new values against the old ones.
df['weightclass'] = df['Fight_type'].map(weight_dict)
df['gender'] = df['Fight_type'].map(gender_dict)
df['title_bout'] = df['Fight_type'].map(title_dict)

## Finally, I update the weightclass to indicate if it was a male or female 
## weightclass to avoid confusion.
df.loc[df['gender'] == 'women', 'weightclass'] = df['weightclass'] + " (w)"

# Here, I identify the columns for the blue and red fighters, respectively and 
# then create two dataframes for Red and Blue Fighters.
b_cols = [col for col in df if col.startswith('B_')]
b_cols.extend(['date', 'weightclass', 'title_bout'])

r_cols = [col for col in df if col.startswith('R_')]
r_cols.extend(['date', 'weightclass', 'title_bout'])

df_b = df[b_cols].copy()
df_r = df[r_cols].copy()

# I drop all columns in those dataframes that were precalculated percentages 
# and then remove the R_ and B_ prefixes from the columns names.
for x in df_b, df_r:
    for col in x.columns:
        if '_pct' in col:
            del x[col]
        elif col != "date" and col != 'weightclass' and col != 'title_bout':
            new_name = col[2:]
            x[new_name] = x[col]
            del x[col]

# Now I create an indicator for champions. Champions are fighting out of the
# red corner in title bouts.
df_r['champion'] = (df_r['title_bout'] == 1)*1
df_b['champion'] = 0
# Now, I append the two columns to get a dataframe consisting of all fights a 
# fighter has had in the UFC and their respective attributes.
df_all = df_b.copy()
df_all = df_all.append(df_r)
df_all.columns


# ### Retaining Fighters with at least three fights  
# Now, with the new long dataframe, I am going to split the attack columns into attempts and successes. When fighters had no attempts of a given type, their values are filled with zeros. Then, I will reduce the width of the dataframe by only keeping features that are relevant to the calculations and comparisons.

# In[ ]:


# Here, I identify the columns that were written as 'x of y' and create new 
# columns containing the successful attempts and total attempts for each type
# of attack.
split_cols = ['TD', 'DISTANCE', 'CLINCH','GROUND']

for col in df_all:
    if col in split_cols:
        var_succ = col + '_succ'
        var_att = col + '_att'        
        df_all[[var_succ, var_att]] = df_all[col].str.split(" of ", expand = True).astype(int)
        del df_all[col]

keep_cols = []
for i in split_cols:
    keep_cols.append(i + "_succ")
    keep_cols.append(i + "_att")
keep_cols.extend(['fighter', 'date', 'win', 'weightclass'])

# This part creates a new dataframe that only includes the columns of interest.
df_all = df_all[keep_cols]
df_all = df_all.sort_values(by = ['fighter', 'date'])
df_all = df_all.fillna(0)

# Calculate the total number of fights a fighter had in his or her career.
df_all['fights'] = 1
df_all['total_fights'] = df_all.groupby('fighter')['fights'].transform('sum')

# Now, I restrict the dataframe to only include fighters who have had a 
# minimum of three fights.
df_all = df_all[df_all['total_fights'] >= 3].copy()


print('The length before restriction was: {},'.format(len(df)*2), 'and the length after restriction is: {}'.format(len(df_all)))


# # Time to Calculate Fighter Efficiency  

# ### Unadjusted Efficiency  
# First, I will calculate all fighters' unadjusted efficiency as:  
# ***Fighter Efficiency = ((Takedowns + Distance Strikes + Clinch Strikes + Ground Strikes) / (Takedown Attempts + Distance Attempts + Clinch Attempts + Ground Attempts)) x 100 ***  
# 
# Naturally, this is a weighted average. So, fighters' total efficiency will be affected by the proportion of their attacks of a certain type. So, fighters who attempt more takedowns will have their total efficiency more influenced by takedown attempts compared to a fighter who attempts few takedowns, for example.  
# 
# Then, I will identify the proportion of each attack type of the total attacks for each fighter, and will also identify the year a fighter entered into the UFC.  
# 
# After that, I no longer need all of these rows, and will only keep one row per fighter.
# 

# In[ ]:


# Here I calculate the total number of attempts and successes for each attack.
# And then I calculate the overal success rate for that type.
att_cols = []
succ_cols = []
for col in df_all.columns:
    if col.endswith('_succ') and ~col.endswith('total_succ'):
        root = col[:-5]
        match = root + '_att'
        total_att = root + '_total_att'
        total_succ = root + '_total_succ'
        df_all[total_att] = df_all.groupby('fighter')[match].transform('sum')
        df_all[total_succ] = df_all.groupby('fighter')[col].transform('sum')        
        df_all[root] = df_all[total_succ] / df_all[total_att] * 100
        att_cols.append(total_att)
        succ_cols.append(total_succ)

# Then, I sum across all attack types and calculate the overall unadjusted rate.
df_all['total_attacks'] = df_all[att_cols].sum(axis = 1)
df_all['total_succ'] = df_all[succ_cols].sum(axis = 1)
df_all['unadjusted'] = df_all['total_succ']/df_all['total_attacks'] * 100

# Here, I calculate the weights of each input to the final average.
for col in att_cols:
    own_wt = col[:-10] + "_own_wt"
    df_all[own_wt] = df_all[col] / df_all['total_attacks']

# Identify each fighter's entry year in the UFC.
df_all['year'] = df_all['date'].str[-4:].astype(int)
df_all['entry_year'] = df_all.groupby('fighter')['year'].transform('min')

# Then, I keep just one row per fighter.
df_unwt = df_all.groupby('fighter').first()
df_unwt = df_unwt.reset_index().copy()
df_unwt = df_unwt.fillna(0)


# # How have fighting styles changed over the history of the UFC?  
# We see that there have been pretty significant changes in how fighters approach the fightgame since the UFC began.  In the early days of the UFC, the ground game made up a big part of fighters' styles. In the early 1990s, nearly 40% of attacks were ground attacks. But as the game has evolved, fighters have increasingly become reliant on distance striking. Takedowns, have never dominated at any point, but this is to be expected, as you need way fewer takedowns to accomplish your ultimate goal of finishing a fight. Just one could be enough. The proportion of fighters incorporating clinch strikes has remained fairly constant since the 1990s.  
# 
# Just from this graph we can see that comparing efficiency across time has some problems, since fighting styles have changed pretty significantly over the past few decades.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')

# Plot the evolution of fighting styles over time.
wgts_cols = [col for col in df_unwt if col.endswith('own_wt')]
yearly = pd.DataFrame(df_unwt.groupby('entry_year')[wgts_cols].mean())
yearly = yearly.reset_index()

colors = ['midnightblue', 'tab:blue', 'mediumpurple', 'indianred']

fig, ax = plt.subplots(figsize = [9,6])
labels = ['Takedowns', 'Distance', 'Clinch', 'Ground']
ax.stackplot(yearly['entry_year'], yearly['TD_own_wt'], yearly['DISTANCE_own_wt'], yearly['CLINCH_own_wt'], yearly['GROUND_own_wt'], labels = labels, colors = colors, edgecolor = 'w', linewidth = 0)
ax.set_xlim(yearly['entry_year'].min(), yearly['entry_year'].max())
ax.set_ylim(0,1)
ax.set_ylabel('Proportion of Attacks')
fig.suptitle('Distribution of Attack Frequencies by\nYear Fighter Entered the UFC')
fig.subplots_adjust(bottom = 0.10, top = .90)
fig.legend(loc = 'lower center', ncol = 4, edgecolor = 'w', facecolor = 'w')


# # How does fighting style affect a fighter's total efficiency?  
# Now, let's plot a fighter's total efficiency against the proportion of total attacks for each attack type to get an idea of how total efficiency may be affected by a fighter's fighting style.  
# 
# Takedowns don't seem to be related all too much to total efficiency, but like I mentioned earlier, takedowns only make up a small portion of the game in terms of a fighter's total attacks.  For the three striking categories, there is a pretty clear relationship between efficiency and how much a fighter commits to one of those categories. For distance strikes, fighters who attempt a greater proportion of their attacks from range tend to also be less efficient. There is a fairly strong negative correlation ($\rho$ = -0.53) between the proporiton of distance strikes and total efficiency. That isn't too surprising, because these attacks will also have a greater volume to compensate for the loss of efficiency and they may also be a bit 'safer', since it will be harder for a fighter to get badly hurt or submitted from distance vs. in the clinch or on the ground.  
# 
# Fighters who attempt more attacks on the ground or in the clinch are far more efficient, but they are also more vulnerable to takedowns or submissions, and we see a moderately strong ***positive*** relationship between efficiency and the proportion of attacks in the clinch ($\rho$ = 0.47) or on the ground ($\rho$ = 0.39).  
# 
# So it is pretty clear that if we want to compare fighters, we need to adjust how we calculate efficiency by accounting for fighting style.

# In[ ]:


from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Creates a subplot of scatterplots for all unweighted average success rates against proportion of attacks
brg = cm.get_cmap('gist_stern_r', 256)
new_colors = brg(np.linspace(0, 1, 256))
new_colors = new_colors[-65:-10]
newcmp = ListedColormap(new_colors)

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = [8, 6])
fig.suptitle("Comparison of Unadjusted Success Rates by \nProportion of Attacks of Different Types")
fig.tight_layout()
fig.subplots_adjust(top = 0.85)

for i in ax.flat:
    i.set(xlabel = "Proportion of Attacks", ylabel = 'Unadjusted Efficiency')
    i.label_outer()
    
attack_cols = [col for col in df_unwt if col.endswith("_own_wt")]
for att in attack_cols:
    newvar = att[:-7] + "_corr"    
    newvar = df_unwt[[att, 'unadjusted']].corr()

y = 0
x = 0
for col in attack_cols:
    ax[y,x].scatter(df_unwt[col], df_unwt['unadjusted'], cmap = newcmp, c = df_unwt['unadjusted'], s = 5)
    root = col[:-7]
    corr = root + "_corr"
    corr = df_unwt[[col, 'unadjusted']].corr()
    z = corr.iloc[0, 1].round(2)
    if root == 'TD':
        new_title = 'Takedowns'
    else:
        new_title = root.title() + " Strikes"
    if root == 'DISTANCE':
        x_pos = 0.1
        y_pos = 20
        ax[y,x].text(x_pos, y_pos, u'$\it\u03c1$ = {: .2f}'.format(z))
    else:
        x_pos = 0.6
        y_pos = 20
        ax[y,x].text(x_pos, y_pos, u'$\it\u03c1$ = {: .2f}'.format(z))
    ax[y,x].set_title(new_title)
    if x == 1:
        y += 1
        x -= 1
    else:
        x += 1


# # Calculating Adjusted Efficiency  
# To account for the differences in fighting styles, I am going to basically 'reweight' the contribution of each fighter's attempts from each attack type to the same standard.  In this case, I am going to reweight the number of attempts by the median proportion of those attempts for all fighters who ever fought at least three fights on UFC roster.  So, in this case, the median proportion of attacks from Distance were 0.780193. If a fighter was less prone to distance striking, and only attempted 50% of his attacks from distance, his new adjusted efficiency would be reweighted with 0.780193 instead of 0.50, etc. After reweighting each attack success rate, these are then summed up to give us the fighters' adjusted efficiency.

# In[ ]:


# Now, I calculate the median number of attempts for all fighters for each 
# attack.
for col in att_cols:
    newvar_att = col[:-10] + "_median_att"
    df_unwt[newvar_att] = df_unwt[col].median()

# And then I sum all the median attacks.
median_att_cols = [col for col in df_unwt if col.endswith('_median_att')]
df_unwt['total_median_att'] = df_unwt[median_att_cols].sum(axis = 1)

# This allows me to construct an artificial weight variable for each attack
# type for the 'median' fighter.
for col in median_att_cols:
    wt_var = col[:-11] + "_wt"
    att_var = col[:-11] + '_median_att'
    df_unwt[wt_var] = df_unwt[att_var]/df_unwt['total_median_att']

# Then, I can reclaculate the rates and weight them by the standard population.   
cols = ['TD_total_succ', 'DISTANCE_total_succ', 'CLINCH_total_succ', 'GROUND_total_succ']
rw_cols = []
for col in cols:
    root = col[:-5]
    rate = root + "_rate"
    wt_var = col[:-11] + "_wt"
    att = root + "_att"
    df_unwt[rate] = df_unwt[col] / df_unwt[att] * df_unwt[wt_var] * 100
    rw_cols.append(rate)


# Finally, I sum across those weighted inputs to arrive at the adjusted 
# overall rate.
df_unwt['adjusted'] = df_unwt[rw_cols].sum(axis = 1)

# Now create a final dataframe with columns of interest.
keep_cols = [col for col in df_unwt if col in 
             ['fighter', 'total_fights', 'win_pct', 'weightclass', 'entry_year', 'adjusted', 'unadjusted'] 
            or col.endswith('_wt')]

df_final = df_unwt[keep_cols].copy()

# Calculate the difference between unadjusted and adjusted efficiency rates
df_final['diff'] = df_final['unadjusted'] - df_final['adjusted']
df_final = df_final[df_final['diff'] < 20]


# # Comparing Unadjusted and Adjusted Attack Efficiency  
# We see that the unadjusted efficiency rates are fairly close to the adjusted ones, but there is a pretty significant amount of variation. The median difference in unadjusted versus adjusted efficiency was only 0.7 percentage points, but there is a decent spread here. At the extremes, we see that one fighters' unadjusted efficiency was overestimated by about 10 percentage points and another's was underestimated by nearly 20 percentage points!

# In[ ]:


import matplotlib.patches as mpatches

## Creating a custom colormap
brg = cm.get_cmap('gist_stern_r', 256)
new_colors = brg(np.linspace(0, 1, 256))
new_colors = new_colors[-65:-10]
newcmp = ListedColormap(new_colors)

## Now the plot
fig, ax = plt.subplots(1, 2, sharex = True, figsize = [10, 6], gridspec_kw = {'width_ratios':[1,1.1]})
unadj = mpatches.Patch(color = 'r', label = 'Unadjusted')
adj = mpatches.Patch(color = 'b', label = 'Adjusted')
fig.tight_layout()
fig.suptitle('Comparison of Unadjusted and Adjusted Attack Efficiency')
fig.subplots_adjust(top = 0.93, bottom = 0.135, wspace = .18)

a = round(df_final['diff'].min(), 1)
b = df_final['diff'].quantile(.25).round(1)
c = round(df_final['diff'].median(), 1)
d = df_final['diff'].quantile(.75).round(1)
e = round(df_final['diff'].max(), 1)

ax[0].scatter(df_final['unadjusted'], df_final['adjusted'], s = 7, cmap = newcmp, c = df_final['adjusted'])
ax[0].plot([0, 80], [0, 80], c = 'k', linewidth = 2)
ax[0].set_xlabel('Unadjusted Efficiency (%)')
ax[0].set_ylabel('Standardized Efficiency (%)')
ax[0].text(0, 82, r'Summary of Differences:', fontweight = 'bold')
ax[0].text(3, 79, r'Min = {}'.format(a))
ax[0].text(3, 75, r'25$^{{th}}$ = {}'.format(b))
ax[0].text(3, 71, r'50$^{{th}}$ =  {}'.format(c))
ax[0].text(3, 67, r'75$^{{th}}$ =  {}'.format(d))
ax[0].text(3, 64, r'Max =  {}'.format(e))

ax[1].hist(df_final['unadjusted'], color = 'r', bins = 40)
ax[1].hist(df_final['adjusted'], color = 'b', bins = 40)
ax[1].set_xlabel('Percent')
ax[1].set_ylabel('Number of Fighters')

ax[1].legend(handles = [unadj, adj], loc = 'upper right', ncol = 1, edgecolor = 'w', facecolor = 'w')


# # How do the current champions stack up against the rest?  
# On the whole, the champions have adjusted efficiencies above the median efficiency for the UFC, but they are by no means the most highly efficient fighters in terms of attack accuracy. They do seem to combine attack styles in a way, though, that can make them more successful overall. Most of the current champions do not rely too heavily on distance striking (most were at or below the median for proportion of distance attacks). And most of them place a greater emphasis on ground attacks than the median fighter, which probably allows them to finish more fights. When it comes to clinch attacks and takedowns, though, the current champions tend to be fairly typical in terms of how much emphasis they place on those attacks.

# In[ ]:


#Plotting the champions against all others for all categories
champs = ['Stipe Miocic', 'Jon Jones', 'Israel Adesanya',
          'Kamaru Usman', 'Khabib Nurmagomedov', 'Max Holloway',
          'Henry Cejudo', 'Amanda Nunes', 'Valentina Shevchenko',
          'Weili Zhang']        

champ_df = df_final[df_final['fighter'].isin(champs)]

wgts_cols = [col for col in df_final if col.endswith('own_wt')]

fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = [8, 6])
fig.suptitle("Fighter Efficiency by Proportion of Attacks of Different Types")
fig.tight_layout()
other = mpatches.Patch(color = 'grey', label = 'Other Fighters')
champions = mpatches.Patch(color = 'orangered', label = 'Current Champions')
fig.subplots_adjust(top = 0.90, bottom = 0.145)

for i in ax.flat:
    i.set(ylabel = "Adjusted Efficiency", xlabel = 'Proportion of All Attacks')
    i.label_outer()

y = 0
x = 0
for col in wgts_cols:
    ax[y,x].scatter(df_final[col], df_final['adjusted'], s = 5, alpha = 0.5, c = 'grey')
    ax[y,x].scatter(champ_df[col], champ_df['adjusted'], s = 18, c = 'orangered')
    median_y_var = df_final['adjusted'].median()
    median_x_var = df_final[col].median()
    median_line = ax[y,x].axhline(y = median_y_var, color = 'k', linestyle = '--', label = 'Median', linewidth = 0.8)
    ax[y,x].axvline(x = median_x_var, color = 'k', linestyle = '--', linewidth = 0.8)
    if col[:-7] == 'TD':
        new_title = 'Takedowns'
    else:
        new_title = col[:-7].title() + " Strikes"
    ax[y,x].set_title(new_title)
    if x == 1:
        y += 1
        x -= 1
    else:
        x += 1

fig.legend(handles = [other, champions, median_line], loc = 'lower center', ncol = 3, edgecolor = 'w', facecolor = 'w')


# # How important is efficiency in different weightclasses?
# Here, I wanted to see how the champions compare to their respective weightclasses in terms of adjusted efficiency.  Basically, I wanted to see if champions are more efficient versus the rest of their division and if this is related to the weight they fight at. The challenge here, is that women have fewer weight classes, with the top weight class being Featherweight. So even though featherweight is one of the smaller weight classes for men, it is the heaviest for women.  So I reconstructed a new relative weight class measure that takes gender into account, so that I can get an idea when comparing ALL fighters, how the gap between champions and the rest of the weight class varies according to weight.  
# 
# The results were pretty interesting. It seems that the heavier the weight class, the larger the efficiency gap is between the champions and the rest.  For the smaller divisions (flyweight, bantamweight, and featherweight) there is basically no difference in the adjuste efficiency of the champions and median fighter. But when we get up to Light Heavyweight, we see that Jon Jones is almost 30% more efficient than the median of the division. This might be because of the inherent differences in conditioning between the weight classes.  Typically, the heavier fighters are more powerful, but have less stamina than lighter fighters. In that case, there is a premium to being highly efficient.  For the lighter weight classes, higher stamina allows even good fighters the opportunity to be less efficient but to have a higher output. And we can expect that the opponents of the champions in the smaller weight classes may also be more mobile (and therefore harder to land on), which could play a role in these numbers too. 

# In[ ]:


## Puts Max Holloway in Correct Weight Class because, although he has fought at lightweight a lot
## he is currently the featherweight champion.
df_final.loc[df_final['fighter'].str.contains('Max Holloway'), 'weightclass'] = 'Featherweight'
## Creates duplicate rows for Amanda Nunes and Henry Cejudo and adds an additional
## weight class.
henry = df_final['fighter'] == 'Henry Cejudo'
df_try = df_final[henry] 
df_final = df_final.append([df_try], ignore_index = True)
df_final.iloc[-1, df_final.columns.get_loc('weightclass')] = 'Bantamweight'

amanda = df_final['fighter'] == 'Amanda Nunes'
df_try = df_final[amanda] 
df_final = df_final.append([df_try], ignore_index = True)
df_final.iloc[-1, df_final.columns.get_loc('weightclass')] = 'Featherweight (w)'

# This block of code will create a new dataframe that only includes the champions' stats
# and the median adjusted efficiency of their entire weight class (excluding themselves)
j=0
for val in set(df_final['weightclass']):
    temp_df = df_final[['weightclass', 'fighter', 'adjusted']][df_final['weightclass'] == val].copy()
    temp_df['champ'] = 0
    for c in champs:
        if c in temp_df['fighter'].values:
            temp_df['champ'] = (temp_df['fighter'] == c)*1
    temp_df = temp_df.sort_values(by = ['champ'], ascending = False)
    temp_df['median'] = temp_df.iloc[1:]['adjusted'].median()
    if j == 0:
        champ_wc_df = temp_df.head(1).copy()
        j += 1
    else:
        champ_wc_df = champ_wc_df.append(temp_df.head(1))

# Here I will remove the weightclasses that aren't true classes (i.e Open weight, catch weight)
champ_wc_df = champ_wc_df[champ_wc_df['champ'] == 1]
champ_wc_df = champ_wc_df.sort_values(by = 'adjusted', ascending = False)

# Here I will create a dictionary of the weight classes and their weights in lbs and then
# create a new column in the dataframe by mapping the dictionary to the weightclass column.
wc_list = [col for col in champ_wc_df['weightclass'].values]
lbsranks = [205, 185, 135, 145, 225, 145, 125, 170, 115, 135, 125, 155]
lbs_dict = dict(zip(wc_list, lbsranks))
champ_wc_df['wc_lbs'] = champ_wc_df['weightclass'].map(lbs_dict)

# Calculate the difference between the champion's adjusted efficiency and the median for the weight class.
champ_wc_df['difference'] = ((champ_wc_df['adjusted'] - champ_wc_df['median']) / champ_wc_df['median'])*100

# Create a list of mens and women's weight classes
womens_wc = [col for col in champ_wc_df['weightclass'].values if col.endswith('(w)')]
mens_wc = [col for col in champ_wc_df['weightclass'].values if col not in womens_wc]

# Calculates a new relative weight class that applies to both genders.
champ_wc_df['relative_wc'] = champ_wc_df['wc_lbs'][champ_wc_df['weightclass'].isin(mens_wc)] / champ_wc_df['wc_lbs'][champ_wc_df['weightclass'].isin(mens_wc)].median() 
champ_wc_df['relative_wc'] = champ_wc_df['relative_wc'].fillna(champ_wc_df['wc_lbs'] / champ_wc_df['wc_lbs'][champ_wc_df['weightclass'].isin(womens_wc)].median())

# Create a list of abbreviations that will be placed on the plot
wc_abbrevs = ['LH', 'MW', 'BW(w)', 'FW(w)', 'HW', 'FW', 'FlW(w)', 'WW', 'SW(w)', 'BW', 'FlW', 'LW']

fig, ax = plt.subplots(figsize = [10, 6])
ax.scatter(champ_wc_df['relative_wc'], champ_wc_df['difference'], cmap = newcmp, c = champ_wc_df['relative_wc'])

for i, txt in enumerate(wc_abbrevs):
    ax.annotate(txt, (champ_wc_df['relative_wc'].iat[i], champ_wc_df['difference'].iat[i]), 
                xytext = (5, 0),textcoords = 'offset points')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

text_str1 = '        '.join(('SW = Strawweight', 'WW = Welterweight'))
text_str2 = '            '.join(('FlW = Flyweight', 'MW = Middleweight'))
text_str3 = '     '.join(('BW = Bantamweight', 'LH   = Light Heavyweight'))
text_str4 = '     '.join(('FW = Featherweight', 'HW  = Heavyweight'))
text_str5 = '          '.join(('LW = Lightweight', "(w)   = Women's Division"))

text_str = '\n'.join((text_str1, text_str2, text_str3, text_str4, text_str5))

props = dict(boxstyle = 'round', facecolor = 'silver', alpha = 0.5)

ax.text(0.6, 0.15, text_str, transform = ax.transAxes, fontsize = 9, bbox = props)
ax.set_ylabel('Percentage Differences vs. Median Adjusted Efficiency')
ax.set_xlabel('Weightclass Relative to Median Weight for Gender')
ax.set_title('Percentage Difference between Champion and Weightclass Efficiency', fontsize = 16)


# # Conclusion
# Well I hope you enjoyed this exploration with me. I was surprised to see that fighter efficiency depended so much on fighters' styles and that the importance of adjusted efficiency seems to be greater for heavier fighters than for smaller ones.  With this simple metric it is easy to compare fighters from different eras, genders, weightclasses, and styles. Even though efficiency clearly isn't everything, it is a fun talking point when chatting with your friends about who is a better fighter.  :-P
