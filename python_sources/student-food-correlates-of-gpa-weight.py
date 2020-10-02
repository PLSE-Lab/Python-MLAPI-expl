#!/usr/bin/env python
# coding: utf-8

# # Correlates of GPA & Weight

# ## Abstract
# I want to analyse numerical x numerical relationships with GPA and Weight. What sort of food-related behaviours predict these two variables of interest?
# 
# ## Findings
# 
#  - Generally found low to weak correlations (r < 0.3) that are statistically significant (p < 0.05)
#  - Higher consumption rates of Thai, Indian, and other ethnic food is positively correlated with GPA scores.
#  - Perception of calorie load in (Panera Bread Roasted Turkey and Avocado BLT) positively predicts GPA, while perception of calorie load in (scones from Starbucks) negatively predicts GPA.
#  - Student parents cooking less frequency is associated with higher GPA scores.
#  - Higher frequency of fruit and vegetable consumption is associated with smaller weight measurements.
# 
# ## Pending Actions
# 
#  - Interpret non-significant correlations with relevant variables.

# ----------
# 
# 
# # Set Up Dataset & Feature Classification

# In[ ]:


from pandas import read_csv
data = read_csv("../input/food_coded.csv")


# In[ ]:


features = data.columns


# In[ ]:


features_by_dtype = {}
for f in features:
    dtype = str(data[f].dtype)
    
    if dtype not in features_by_dtype.keys():
        features_by_dtype[dtype] = [f]
    else:
        features_by_dtype[dtype] += [f]


# In[ ]:


def code(value, code_dictionary):
    if value in code_dictionary.keys():
        return code_dictionary[value]
    else:
        return value
    
def ordinalizer(data,feature):
    output = {}
    unique = sorted(data[feature].unique().tolist())
    j = 1
    for i in [i for i in unique if str(i) != "nan"]:
        output[i] = j
        j += 1
    
    return output
    
def code_features_as_ordinal(data, to_be_coded):
    
    for feature in to_be_coded:
        cd = ordinalizer(data,feature)
        data[feature] = data[feature].apply(code, code_dictionary=cd)


# In[ ]:


food_calorie_perception = [i for i in features_by_dtype["int64"] if "calories" in i]
food_calorie_perception += ["calories_scone", "tortilla_calories"]


# In[ ]:


code_features_as_ordinal(data,food_calorie_perception)


# In[ ]:


discrete_features = [i for i in features_by_dtype["int64"] if (not "calories" in i) & (data[i].unique().size > 2) & ("coded" not in i)]
discrete_features += ["calories_day","cook","exercise","father_education","income","life_rewarding","mother_education","persian_food"]


# In[ ]:


remove = [15,102,104,2,32,74,61]
data = data.drop(remove,0)

data.loc[67,"weight"] = "144"
data.loc[3, "weight"]= "240"
data["weight"] = data["weight"].apply(int)

data.loc[73,"GPA"] = 3.79
data["GPA"] = data["GPA"].apply(float)


# ## Selected Numerical Variables

# In[ ]:


data[discrete_features + food_calorie_perception + ["GPA", "weight"]].head()


# In[ ]:


discrete_features + food_calorie_perception + ["GPA", "weight"]


# In[ ]:


from scipy.stats import pearsonr,spearmanr,kendalltau
from pandas import DataFrame

def correlation_table(data,numerical_features,target):
    
    rows_list = []

    for x2 in numerical_features:
    
        x1 = target
        
        row = {}
        row["Variable A"] = x1 
        row["Variable B"] = x2
        
        x3 = data[data[x1].notnull() & data[x2].notnull()]

        pearson = pearsonr(x3[x1],x3[x2])
        row["Pearson"] = pearson[0]
        row["Pearson's p-value"] = pearson[1]

        spearman = spearmanr(x3[x1],x3[x2])
        row["Spearman"] = spearman[0]
        row["Spearman's p-value"] = spearman[1]

        kendall = kendalltau(x3[x1],x3[x2])
        row["Kendall"] = kendall[0]
        row["Kendall's p-value"] = kendall[1]

        rows_list.append(row)

    ordered_columns = ["Variable A", "Variable B", "Pearson", "Pearson's p-value", "Spearman", "Spearman's p-value", "Kendall", "Kendall's p-value"]
    
    corr = DataFrame(rows_list, columns=ordered_columns).round(2)
    corr = corr[(corr["Variable A"] == target) | (corr["Variable B"] == target)]
    return corr


# In[ ]:


from seaborn import regplot, set_style
from matplotlib.pyplot import show, suptitle
from scipy.stats import linregress
from IPython.display import display

def display_nxn_analysis(data,feature,target):
    
    # === Reg Plot === #
    
    set_style("whitegrid")
    ax = regplot(data=data,x=feature, y=target, marker="x", scatter_kws={"s": 50})
    suptitle(target + " x " + feature)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    show()

    # === Correlations === #
    
    print("Correlation Scores")
    corr = correlation_table(data,[feature],target)
    display(corr)
    
    # === Simple Linear Regression === #
    
    print("Simple Linear Regression")
    no_nulls = data[data[feature].notnull() & data[target].notnull()]
    slope, intercept, r_value, p_value, std_err = linregress(no_nulls[feature],no_nulls[target])
    display(DataFrame([{"R^2" : r_value, "standard error" : std_err, "p-value" : p_value,}]).round(2))


# ----------
# 
# 
# # GPA 

# ## 1. Statistically Significant Correlations Table

# In[ ]:


GPA_corr = correlation_table(data,discrete_features+food_calorie_perception,"GPA")
pearson_p = GPA_corr["Pearson's p-value"] < 0.05
spearman_p = GPA_corr["Spearman's p-value"] < 0.05
kendall_p = GPA_corr["Kendall's p-value"] < 0.05
GPA_corr = GPA_corr.loc[pearson_p | spearman_p | kendall_p]
GPA_corr


# ## 2. Break Down of Each GPA x Numeric Relationship 

# In[ ]:


numeric = iter(GPA_corr["Variable B"].tolist())


# # Student's frequency of consuming ethnic food positively predicts GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# # Higher grade levels are positively associated with higher GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# # Students' consumption of Indian food is positively associated with GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# # Students' parents cooking less frequency is associated with higher GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# # When available, students' likelihood of eating Thai cuisine is positively associated with GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# # Students perception of higher calorie load from (Panera Bread Roasted Turkey and Avocado BLT) is positively associated with GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# # Student's perception of higher calorie load in scones from Starbucks is negatively associated with GPA scores.

# In[ ]:


display_nxn_analysis(data,next(numeric),"GPA")


# ----------
# 
# 
# # Weight

# In[ ]:


weight_corr = correlation_table(data,discrete_features+food_calorie_perception,"weight")
pearson_p = weight_corr["Pearson's p-value"] < 0.05
spearman_p = weight_corr["Spearman's p-value"] < 0.05
kendall_p = weight_corr["Kendall's p-value"] < 0.05
weight_corr = weight_corr.loc[pearson_p | spearman_p | kendall_p]
weight_corr


# In[ ]:


numeric = iter(weight_corr["Variable B"].tolist())


# # More frequent consumption of fruits per a day is negatively associated with weight.

# In[ ]:


display_nxn_analysis(data,next(numeric),"weight")


# # Higher grade levels is associated with heavier weight measurements.

# In[ ]:


display_nxn_analysis(data,next(numeric),"weight")


# # More frequent consumption of veggies per a day is associated with smaller weight measurements.

# In[ ]:


display_nxn_analysis(data,next(numeric),"weight")

