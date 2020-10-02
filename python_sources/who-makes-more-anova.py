"""
Let's try to see if there is notable differences in the wages of people according to their mean of transportation
to work. This looks like an ANOVA (analysis of variance) study case, so this is what I'll be trying to do!

I will run some basic normality and homoscedasticity tests before non parametric ANOVA (the Kruskal Wallis test) in
order to see if some distributions are statistically different (median wise for the non parametric case).
Then I run Mann Whitney tests (a non parametric version of a t test) to find out which weights distributions 
really differ by their medians.

This is my first script and it was an opportunity to learn more about these different statistical tests to assess
a "fun" observation. Please feel free to comment or suggest any improvement, that's what I'm here for!
"""

import itertools

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy import stats

columns = ['PUMA', 'ST', 'AGEP', 'SEX', 'WAGP', 'JWTR']

# the exemple is ran for colorado state
ST = 8  # state code
PUMA = None  # 703, 1600, 100, 3302, 300  
SEX = 1 # 1 MALE, 2 FEMALE
#################
# load dataframes
# I need to do this by chunks, my laptop won't hold otherwise..

dfa = pd.DataFrame(columns=columns)
chunka = pd.read_csv('../input/pums//ss13pusa.csv', chunksize=1000, usecols=columns)

while True:
    try:
        sub_df = chunka.get_chunk()

        sub_df = sub_df[(sub_df['ST'] == ST) & (sub_df['SEX'] == SEX)]
        
        if PUMA:
            sub_df = sub_df[sub_df['PUMA'] == PUMA]
            
        sub_df = sub_df.dropna()

        dfa = pd.concat([sub_df, dfa])
    except:
        break
    
dfb = pd.DataFrame(columns=columns)

chunkb = pd.read_csv('../input/pums//ss13pusb.csv', chunksize=1000, usecols=columns)

while True:
    try:
        sub_df = chunkb.get_chunk()

        sub_df = sub_df[(sub_df['ST'] == ST)  & (sub_df['SEX'] == SEX)]
        if PUMA:
            sub_df = sub_df[sub_df['PUMA'] == PUMA]

        sub_df = sub_df.dropna()

        dfb = pd.concat([sub_df, dfb])
    except:
        break
    
df = pd.concat([dfa, dfb])
print("n observations: ", len(df))

# use modalities expressions
modalities_dict = {1: "car",
                   2: "Bus",
                   3: "Streetcar",
                   4: "subway",
                   5: "Railroad",
                   6: "Ferryboat",
                   7: "Taxicab",
                   8: "Motorcycle",
                   9: "Bicycle",
                   10: "Walked",
                   11: "Worked at home",
                   12: "Other method"}

df['JWTR'] = [modalities_dict[i] for i in df['JWTR'].values]

# store in dict for faster access
vars_dict = {}

for modality in df['JWTR'].unique():
    mask = df['JWTR'] == modality
    if sum(mask) > 100:
        vars_dict[modality] = list(df[mask]['WAGP'].values)
        
all_modalities = list(vars_dict.keys())

print("number of present transportations: ", len(all_modalities))

#################################
# distributions of weights per transportation mean
# thanks to kati for the code idea

plt.figure()

bp = df[df['JWTR'].isin(all_modalities)].boxplot(column="WAGP", by="JWTR")

plt.ylabel("wages ($)")
bp.set_ylim([-50, 300000])
plt.xticks(range(len(all_modalities)), list(all_modalities), rotation=90)
plt.xlabel('')

plt.title("boxplot of wages per transportation mean")

plt.savefig("boxplot_of_wages.png")

#####################
# let's test normality: Shapiro Test

vars_table = {}

print("test normality: Shapiro")
print("modality -- p_value -- reject")
for modality in all_modalities:
    w_stat, p_val = stats.shapiro(vars_dict[modality])
    vars_table[modality] = {'p_value': p_val,
                            'reject': True if p_val < 0.05 else False}
    print("{} -- {} -- {}"
          .format(modality, vars_table[modality]['p_value'], vars_table[modality]['reject']))

###############################
# what about homoscedasticity ? Levene's test with median cut (for heavy tailed distribtions)

l_stat, p_val = stats.levene(*list(vars_dict.values()),
                             center='median')

print("Homoscedasticity test: Levene with median cut")
print("p value:", p_val)

###############
# Kruskal-Wallis test (non parametric ANOVA)

h_stat, p_val = stats.mstats.kruskalwallis(*list(vars_dict.values()))

print("Non parametric ANOVA (Kruskal Wallis)")
print("p value:", p_val)

#####################################
# U tests
# we could use Bonferroni correction to adjust for multiple comparisons


mat = np.zeros([len(all_modalities), len(all_modalities)])
vars_table = {}
# print("modality -- modality -- median -- p_value -- reject")
for modality_1, modality_2 in itertools.product(all_modalities, repeat=2):
    mw_stat, p_val = stats.mannwhitneyu(vars_dict[modality_1], vars_dict[modality_2], use_continuity=False)
    
    vars_table[(modality_1, modality_2)] = {'p_value': p_val,
                            'reject': True if p_val < 0.05 else False,
                            'median_diff': np.median(vars_dict[modality_1]) - np.median(vars_dict[modality_2])}
    # print("{} -- {} -- {} -- {}"
    #       .format(modality_1, modality_2,
    #       vars_table[(modality_1, modality_2)]['median_diff'],
    #       vars_table[(modality_1, modality_2)]['p_value'],
    #       vars_table[(modality_1, modality_2)]['reject']))
    
    mat[list(all_modalities).index(modality_1), list(all_modalities).index(modality_2)] = 1 if p_val < 0.05 else 0

fig = plt.figure()
plt.rcParams['figure.figsize'] = 8, 8

plt.imshow(mat)  # I find imshow cooler than matshow

plt.xticks(range(len(all_modalities)), list(all_modalities), rotation=90)
plt.yticks(range(len(all_modalities)), list(all_modalities), rotation=0)

plt.title('U test result matrix. Blue: no reject H0. red: reject H0, where H0: weight distributions are statistically same', fontdict={'fontsize': 10, 'horizontalalignment': 'center'})

plt.savefig("u_test_transportations.png")

################################################
# plot distributions with most statistically different medians

def plot_xy(modality_1=None, modality_2=None, n_bins=30):

    fig = plt.figure()
    
    plt.hist(vars_dict[modality_1], color='k', label=modality_1, bins=n_bins, normed=1, histtype='step')
    plt.hold(True)
    plt.hist(vars_dict[modality_2], color='r', label=modality_2, bins=n_bins, normed=1, histtype='step')
    plt.legend()
    
    plt.xlabel('wages')
    
    plt.title('histograms of most statistically different wages distributions')
    
    fig.savefig("diff_dist.png")
    
max_modality = max([key for key in vars_table.keys() if vars_table[key]['reject']], key=lambda k: abs(vars_table[k]['median_diff']) )

print("biggest medians difference:: ", vars_table[max_modality]['median_diff'], "for ", max_modality)
plot_xy(*max_modality)