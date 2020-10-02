#!/usr/bin/env python
# coding: utf-8

# # Pokemon Population Distribution Analysis
# ![](https://vignette.wikia.nocookie.net/pokemon/images/3/32/IL012_Im019.png/revision/latest?cb=20141212175012)
# 
# ## Table of Contents:
# * [Introduction](#introduction)
# * [Exploratory Data Analysis](#eda)
#     * i. [Importing Packages](#packages)
#     * ii. [Importing Data](#data)
#     * 1. [Subset Comparisons](#subsets)
#     * 2. [Type Analysis](#types)
#         * 2a. [Type Distributions](#type_dist)
#             * [Type Correlation](#type_corr)
#             * [Generational Type Distributions](#gen_type)
#             * [Legendary Type Distributions](#lgd_type)
#         * 2b. [Primary Type Stat Distributions](#type_stat)
#         * 2c. [Generational Stat Distributions](#gen_stat)
# 
# <a id='introduction'></a>
# # Introduction
# This dataset contains Pokemon from Generations 1 to 6 (National Pokedex numbers 1 to 721). Alternate forms with stat changes introduced after Gen6 in Sun & Moon and Ultra Sun & Ultra Moon are unavailable (*e.g.* Zygarde Complete Forme).
# 
# This dataset also does not contain any information with regards to evolutionary lineage information between the different Pokemon, therefore analysis will move forward with the naive assumption that all Pokemon and their different forms are independent.
# 
# The following analyses aims to visualize the various distributions of stats across types and generations.
# 
# <a id='eda'></a>
# # Exploratory Data Analysis
# <a id='packages'></a>
# ## i. Importing Packages

# In[ ]:


# Analysis
import numpy as np
import pandas as pd
import scipy.stats as sps

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='data'></a>
# ## ii. Importing Data

# In[ ]:


pkmn_df = pd.read_csv("../input/Pokemon.csv")
pkmn_df.info()


# The input data appears to be very clean and processed, the only `NaN` values present are in the `Type 2` columns of single-type Pokemon. Now let's inspect the data entries themselves a bit more.

# In[ ]:


# Replace null values with a more sensible string value.
pkmn_df.fillna("None", inplace = True)
pkmn_df.head(20)


# In[ ]:


pkmn_df.tail(20)


# <a id='subsets'></a>
# ## 1. Subset Comparisons
# From initial inspection of the data, special forms such as Mega Evolutions and Primal Forms are listed as separate entries. Due to the stat changes these forms confer as well as the stat skewing stemming from the inclusion of Legendaries, let's separate special (Mega Evolutions, Primal Forms, Legendaries) Pokemon and non-special Pokemon.

# In[ ]:


non_special_pkmn = pkmn_df[~(pkmn_df.Name.str.contains("Mega|Primal")) & ~(pkmn_df.Legendary)]
special_pkmn = pkmn_df[~pkmn_df.index.isin(non_special_pkmn.index)]

# T-test
non_special_vs_special = sps.ttest_ind(non_special_pkmn.Total, special_pkmn.Total)

print("Number of \"non-special\" Pokemon: \t%d" %(non_special_pkmn.shape[0]))
print("Number of \"special\" Pokemon: \t\t%d" %(special_pkmn.shape[0]))
print("\nNon-Special vs. Special Total stat independent T-test:")
print("T Value: %.2f" %(non_special_vs_special[0]))
print("P Value: %.2e" %(non_special_vs_special[1]))


# There are a lot less "special" Pokemon than non-special Pokemon. Nonetheless, the simple independent T-test of the `Total` stat implies a difference between the two datasets. But what about Mega Evolutions of non-Legendary Pokemon compared with base forms of Legendary Pokemon?

# In[ ]:


non_legend_megas = special_pkmn[~special_pkmn.Legendary]
legendary_pkmn = special_pkmn[special_pkmn.Legendary]

# Hoopa Unbound form provides a stat boost, will be considered as a "mega".
non_mega_legends = special_pkmn[
    ~(special_pkmn.Name.str.contains("Mega|Primal|Unbound")) &
    (special_pkmn.Legendary)
]
mega_legends = special_pkmn[
    ~(special_pkmn.index.isin(non_mega_legends.index)) &
    ~(special_pkmn.index.isin(non_legend_megas.index))
]

# T-tests
non_legend_mega_vs_base_legend = sps.ttest_ind(
    non_legend_megas.Total, non_mega_legends.Total
)
base_legend_vs_mega_legend = sps.ttest_ind(
    non_mega_legends.Total, mega_legends.Total
)

print(
    "Number of non-Legendary Mega Evolutions: \t%d" %(non_legend_megas.shape[0])
)
print("Number of base Legendary Pokemon: \t\t%d" %(non_mega_legends.shape[0]))
print("Number of Legendary Mega Evolutions: \t\t%d" %(mega_legends.shape[0]))
print()
print("Non-Legendary Mega vs. base Legendary Total stat independent T-test:")
print("T Value: %.2f" %(non_legend_mega_vs_base_legend[0]))
print("P Value: %.2f" %(non_legend_mega_vs_base_legend[1]))
print("\nBase Legendary vs Legendary Mega Total stat independent T-test:")
print("T Value: %.2f" %(base_legend_vs_mega_legend[0]))
print("P Value: %.2e" %(base_legend_vs_mega_legend[1]))


# The difference between non-Legendary Mega Evolutions and base Legendaries is much less pronounced, though base Legendaries still generally have higher stats than non-Legendary Megas. Let's visualize the difference between the subsets.

# In[ ]:


kde_figs, (ax_ns_v_s, ax_nlm_v_nml, ax_nl_v_l) =     plt.subplots(3, 1, figsize = [10, 15], sharex = True, sharey = True)

sns.kdeplot(non_special_pkmn.Total, ax = ax_ns_v_s, shade = True)
sns.kdeplot(special_pkmn.Total, ax = ax_ns_v_s, shade = True)

sns.kdeplot(
    non_legend_megas.Total, ax = ax_nlm_v_nml, shade = True,
    color = "green"
)
sns.kdeplot(
    non_mega_legends.Total, ax = ax_nlm_v_nml, shade = True,
    color = "red"
)

sns.kdeplot(
    non_special_pkmn.Total, ax = ax_nl_v_l, shade = True,
    color = "purple"
)
sns.kdeplot(
    legendary_pkmn.Total, ax = ax_nl_v_l, shade = True,
    color = "gold"
)
sns.despine()

ax_ns_v_s.set_title("Non-Special vs. Special", fontsize = 16)
ax_ns_v_s.legend(labels = ["Non-Special", "Special"])
ax_nlm_v_nml.set_title(
    "Non-Legendary Megas vs. Base Legendaries", fontsize = 16
)
ax_nlm_v_nml.legend(labels = ["Non-Legendary Mega", "Base Legendary"])
ax_nl_v_l.set_title("Non-Legendary vs. Legendary", fontsize = 16)
ax_nl_v_l.legend(labels = ["Non-Legendary", "Legendary"])
kde_figs.suptitle(
    "Kernel Density Estimate Comparisons of Different Pokemon Subsets",
    fontsize = 20, y = 0.93
)
kde_figs.text(
    x = 0.5, y = 0.08, s = "Total Stat", ha = "center", fontsize = 16
)
kde_figs.text(
    0.04, 0.5, "Proportion", va = "center", rotation = "vertical",
    fontsize = 16
)


# The seemingly bimodal distribution of base Legendaries matches our prior knowledge of Legendary Pokemon. Within each generation, there are weaker Mythical  duos/trios (*e.g.* birds from Gen1) present alongside the cover Legendaries (*e.g.* Groudon, Kyogre, and Rayquaza from Gen3). The small third mode within the Legendary subset is likely due to the inclusion of Mega/Primal forms of Legendaries.
# 
# <a id='types'></a>
# ## 2. Type Analysis
# <a id='type_dist'></a>
# ### 2a. Type Distributions
# Now let's look at some distributions of the different Pokemon types.

# In[ ]:


# Formatting
type_dist_fig, (ax_pt, ax_st) = plt.subplots(2, 1, figsize = [10, 15], sharex = True)
sns.countplot(
    y = "Type 1", data = pkmn_df,
    order = pkmn_df["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax_pt
)
sns.countplot(
    y = "Type 2", data = pkmn_df,
    order = pkmn_df["Type 2"].value_counts().index,
    palette = "PuBu_d", ax = ax_st
)
sns.despine()

# Labeling
ax_pt.set_title("Primary Type Frequencies", fontsize = 16)
ax_pt.set_xlabel("")
ax_pt.set_ylabel("Primary Type", fontsize = 14)
ax_st.set_title("Secondary Type Frequencies", fontsize = 16)
ax_st.set_xlabel("Count", fontsize = 14)
ax_st.set_ylabel("Secondary Type", fontsize = 14)
type_dist_fig.suptitle("Type Distributions", fontsize = 20, y = 0.93)


# Unsurprisingly, `None` is the most common secondary type. Another noted observation is that `Flying` is the least frequent primary type but the most frequent secondary type (where a secondary type is present).
# 
# `Water` being the most frequent primary type also intuitively makes sense, as within the games there is a geographical separation between land and water routes. Wild Pokemon found on water routes will primarily have `Water` as the primary type, whereas on land routes type distributions are much more diverse.
# 
# Now let's look at the correlation matrix between the types.
# 
# <a id='type_corr'></a>

# In[ ]:


primary_type_encoded = pd.get_dummies(pkmn_df["Type 1"])
primary_type_encoded["None"] = 0
secondary_type_encoded = pd.get_dummies(pkmn_df["Type 2"])
type_corr = (primary_type_encoded + secondary_type_encoded).corr()

# Figure
type_heatmap, ax_h = plt.subplots(figsize = [15, 10])
sns.heatmap(type_corr, cmap = "GnBu", linewidth = 0.01, ax = ax_h)
type_heatmap.suptitle(
    "Type Correlation Matrix", fontsize = 20, x = 0.45, y = 0.93
)


# Some notable associations and examples include:
# * Flying/Normal (starting route birds)
# * Bug/Poison (starting route bugs)
# * Grass/Poison (Oddish)
# * Rock/Ground (Geodude)
# * Bug/Flying (starting route bug final evolutions)
# 
# Flying/None surprisingly enough is the nost notable disassociation. Personally, I was not aware of this. Upon consulting Bulbapedia, it appears that as of Gen7, Tornadus (#641) is the only pure Flying type Pokemon in existence.
# 
# Next, let's look at type distributions by generation.
# <a id='gen_type'></a>

# In[ ]:


# Formatting
gen1_type = pkmn_df[pkmn_df.Generation == 1].iloc[:, 2:4]
gen2_type = pkmn_df[pkmn_df.Generation == 2].iloc[:, 2:4]
gen3_type = pkmn_df[pkmn_df.Generation == 3].iloc[:, 2:4]
gen4_type = pkmn_df[pkmn_df.Generation == 4].iloc[:, 2:4]
gen5_type = pkmn_df[pkmn_df.Generation == 5].iloc[:, 2:4]
gen6_type = pkmn_df[pkmn_df.Generation == 6].iloc[:, 2:4]

type_dist_gen_fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) =     plt.subplots(2, 3, figsize = [30, 20], sharex = True)

sns.countplot(
    y = "Type 1", data = gen1_type,
    order = gen1_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax1
)
sns.countplot(
    y = "Type 1", data = gen2_type,
    order = gen2_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax2
)
sns.countplot(
    y = "Type 1", data = gen3_type,
    order = gen3_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax3
)
sns.countplot(
    y = "Type 1", data = gen4_type,
    order = gen4_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax4
)
sns.countplot(
    y = "Type 1", data = gen5_type,
    order = gen5_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax5
)
sns.countplot(
    y = "Type 1", data = gen6_type,
    order = gen6_type["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax6
)
sns.despine()

# Labeling
ax1.set_title("Gen 1", fontsize = 20)
ax2.set_title("Gen 2", fontsize = 20)
ax3.set_title("Gen 3", fontsize = 20)
ax4.set_title("Gen 4", fontsize = 20)
ax5.set_title("Gen 5", fontsize = 20)
ax6.set_title("Gen 6", fontsize = 20)

ax1.set_xlabel(""); ax1.set_ylabel("")
ax2.set_xlabel(""); ax2.set_ylabel("")
ax3.set_xlabel(""); ax3.set_ylabel("")
ax4.set_xlabel(""); ax4.set_ylabel("")
ax5.set_xlabel(""); ax5.set_ylabel("")
ax6.set_xlabel(""); ax6.set_ylabel("")

type_dist_gen_fig.suptitle(
    "Generational Type Distributions", fontsize = 30, y = 0.93
)
type_dist_gen_fig.text(
    x = 0.5, y = 0.08, s = "Count", ha = "center", fontsize = 25
)
type_dist_gen_fig.text(
    0.08, 0.5, "Primary Type", va = "center", rotation = "vertical",
    fontsize = 25
)


# Interestingly enough, `Poison` was fairly prominent in Gen1 but fell off the radar for new Pokemon in the subsequent generations. The `Normal` type bias from Gen4 and Gen5 is also interesting to me, as I remember mistaking a lot of Pokemon from those games for other types due to their appearance only to find out that they were `Normal` upon capturing them.
# 
# Next, let's take a look at the type distributions of Legendary Pokemon.
# 
# <a id='lgd_type'></a>

# In[ ]:


# Formatting
lgd_type_dist_fig, (ax_lpt, ax_lst) =     plt.subplots(2, 1, figsize = [10, 15], sharex = True)
sns.countplot(
    y = "Type 1", data = legendary_pkmn,
    order = legendary_pkmn["Type 1"].value_counts().index,
    palette = "PuRd_d", ax = ax_lpt
)
sns.countplot(
    y = "Type 2", data = legendary_pkmn,
    order = legendary_pkmn["Type 2"].value_counts().index,
    palette = "PuBu_d", ax = ax_lst
)
sns.despine()

# Labeling
ax_lpt.set_title("Primary Type Frequencies", fontsize = 16)
ax_lpt.set_xlabel("")
ax_lpt.set_ylabel("Primary Type", fontsize = 14)
ax_lst.set_title("Secondary Type Frequencies", fontsize = 16)
ax_lst.set_xlabel("Count", fontsize = 14)
ax_lst.set_ylabel("Secondary Type", fontsize = 14)
lgd_type_dist_fig.suptitle(
    "Legendary Type Distributions", fontsize = 20, y = 0.93
)


# While the prevalence of `Psychic` and `Dragon` within the Legendary pool is unsurprising, I find their ordering is a bit unexpected due to the dominance of `Dragon` types among cover Legendaries. A potential explanation here is that Mythical Pokemon, which are more numerous than cover Legendaries, have many more `Psychic` types among them.
# 
# Now let's look at stat distributions across different primary types.
# 
# <a id='type_stat'></a>
# ### 2b. Primary Type Stat Distributions
# The figure for `Total` is shown. Other stats are present but hidden initially for the sake of presentation within this notebook. Please unhide the outputs to see the stat distributions of other stats across primary types.

# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Total", color = "red", shade = True)
g.fig.suptitle(
    "Total Stat Distribution Across Types", y = 1.01, fontsize = 20
)


# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "HP", color = "green", shade = True)
g.fig.suptitle(
    "HP Distribution Across Types", y = 1.01, fontsize = 20
)


# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Attack", color = "purple", shade = True)
g.fig.suptitle(
    "Attack Distribution Across Types", y = 1.01, fontsize = 20
)


# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Defense", color = "blue", shade = True)
g.fig.suptitle(
    "Defense Distribution Across Types", y = 1.01, fontsize = 20
)


# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Sp. Atk", color = "orange", shade = True)
g.fig.suptitle(
    "Special Attack Distribution Across Types", y = 1.01, fontsize = 20
)


# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Sp. Def", color = "gold", shade = True)
g.fig.suptitle(
    "Special Defense Distribution Across Types", y = 1.01, fontsize = 20
)


# In[ ]:


g = sns.FacetGrid(pkmn_df, col = "Type 1", col_wrap = 3)
g.map(sns.kdeplot, "Speed", color = "brown", shade = True)
g.fig.suptitle(
    "Speed Distribution Across Types", y = 1.01, fontsize = 20
)


# ***Note***: `Flying` stats are skewed due to their low numbers as the primary stat.
# 
# Though not very visually distinct, these visualizations more or less confirms the intuitive associations of types with stats that have been generally accepted by the community over the years:
# * `Electric` and `Flying` types are fast.
# * `Psychic` types serve as exceptional special sweepers, but generally have low physical defense.
# * Similarly, `Fighting` types are great physical sweepers, but also very glass cannonish.
# * `Ice`, `Rock`, and `Ground` types tend to be on the slower side.
# * `Steel` Pokemon are great physical walls.
# * `Dragon` types are great all-arounders (more physical than special), but a tad slow.
# * `Bug` types, outside of certain fringe cases, kinda suck.
# 
# Lastly, let's look a bit more into differences across generations.
# 
# <a id='gen_stat'></a>
# ### 2c. Generational Stat Distributions

# In[ ]:


stat_dist_gen_fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) =     plt.subplots(4, 2, figsize = [40, 20], sharex = True)


sns.violinplot(x = "Generation", y = "Total", data = pkmn_df, ax = ax1)
sns.violinplot(x = "Generation", y = "HP", data = pkmn_df, ax = ax2)
sns.violinplot(x = "Generation", y = "Attack", data = pkmn_df, ax = ax3)
sns.violinplot(x = "Generation", y = "Defense", data = pkmn_df, ax = ax4)
sns.violinplot(x = "Generation", y = "Sp. Atk", data = pkmn_df, ax = ax5)
sns.violinplot(x = "Generation", y = "Sp. Def", data = pkmn_df, ax = ax6)
sns.violinplot(x = "Generation", y = "Speed", data = pkmn_df, ax = ax7)

ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

ax1.set_ylabel("Total", fontsize = 18)
ax2.set_ylabel("HP", fontsize = 18)
ax3.set_ylabel("Attack", fontsize = 18)
ax4.set_ylabel("Defense", fontsize = 18)
ax5.set_ylabel("Sp. Atk", fontsize = 18)
ax6.set_ylabel("Sp. Def", fontsize = 18)
ax7.set_ylabel("Speed", fontsize = 18)

stat_dist_gen_fig.suptitle(
    "Generational Stat Distributions", fontsize = 40,
    y = 0.93
)
stat_dist_gen_fig.text(
    x = 0.5, y = 0.08, s = "Generation", ha = "center", fontsize = 30
)
stat_dist_gen_fig.text(
    0.08, 0.5, "Stat", va = "center", rotation = "vertical",
    fontsize = 30
)

sns.despine()


# Outside of certain outliers, there appears to be no significant visual differences between the generations with regards to the different stats, suggesting that each generation is fairly balanced in terms of the Pokemon released.
# 
# This concludes the various distribution analyses for this dataset. Any feedback/comments are greatly appreciated!
# 
# Next, I will attempt to tackle some prediction problems as posed by the creator of this dataset in a separate kernel (will add link once it is complete). Most notably, the question of whether or not the primary type can be inferred from any two individual stats (outside of the combination of `Attack` and `Defense`, which the dataset author states cannot infer the primary type).
# 
# ## Thank you for reading!
