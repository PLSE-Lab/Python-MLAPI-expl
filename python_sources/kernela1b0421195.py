# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", "{:.2f}".format)

adult_Df = pd.read_csv("C:/Users/HP/Desktop/EDAs/adult.data.csv")

print(adult_Df.info(), '\n')
print(adult_Df.head(10))

# Initiating Checks and Cleaning
## dropping '?'
q_col = []
for col in adult_Df.columns:
    if len(adult_Df[adult_Df[col] == '?'] ) >= 1:
        q_col.append(col)
        adult_Df.drop(adult_Df.index[adult_Df[adult_Df[col] == '?'].index], inplace = True)
        adult_Df = adult_Df.reset_index(drop = True)
print(q_col, '\n')

## Dropping underaged
adult_Df.drop(adult_Df.index[adult_Df[adult_Df['age'] < 18].index], inplace=True)
adult_Df = adult_Df.reset_index(drop=True)
print(adult_Df.info())
# 30,162 from 32,561

## Reclassifying education
PreSchool = ['Preschool']
SecondarySchool = ['HS-grad','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','Some-college']
Tertiary = ['Bachelors', 'Masters', 'Doctorate']
Other = ['Assoc-voc', 'Assoc-acdm', 'Prof-school']

def GetEdu (edu):
    if edu in PreSchool:
        return "PreSchool"
    elif edu in SecondarySchool:
        return "SecondarySchool"
    elif edu in Tertiary:
        return "Tertiary"
    else:
        return "Others"

adult_Df['EduClass'] = adult_Df['education'].apply(lambda cell: GetEdu (cell))

## Saving our clean dataFrame
#adult_Df.to_csv('C/Users/HP/Desktop/EDAs/clean_adult_Data.csv', encoding = 'utf-8', index = False)

# 2. Build Data Profiles, Tables and Plots
drp = ['education-num', 'education', 'relationship']
adult_Df.drop(columns = drp, inplace  = True)

## 1. Histogram
plt.figure(figsize = (20, 10))

cat_col = adult_Df.select_dtypes(include = 'object').columns
#for i, col in enumerate(cat_col):
#    plt.subplot(5, 4, i+1)
#    sns.catplot(adult_Df[col], color = ('xkcd:lime'))
#    plt.title(f"{col}", fontsize = 10)
#plt.tight_layout()
#plt.show()
#hist = adult_Df.hist(bins = 10, figsize = (20, 10))


## 2. Boxplots
plt.figure(figsize=(10, 10))

num_cols = adult_Df.select_dtypes(include = 'int64').columns
for i, col in enumerate(num_cols):
    plt.subplot(3, 2, i+1) #Size, Row, ..
    sns.boxplot(adult_Df[col], color = ('xkcd:lime'))
    plt.title(f"{col}", fontsize = 10)
    plt.xlabel('')
plt.tight_layout()
plt.show()

# In the Box plots we can observe high outliers for captial returns, Hours of work per week is uniquely
# centered in the middle with equally spread outliers.

## 3. PairPlots
#sns.pairplot(adult_Df)
#plt.show()

# Measuring Variables
## 1. CATEGORICAL VARIABLES
print()
n_cat = []
for col in adult_Df.columns:
    if adult_Df[col].dtype == object :
        n_cat.append(col)
        print()
        print(f"{adult_Df[col].value_counts()}")
print(f'no. of CAT Variables: {len(n_cat)}')

## 2. NUMERICAL VARIABLES
print()
n_num = []
for col in adult_Df.columns:
    if adult_Df[col].dtype != object and len(adult_Df[col].unique()) < 25:
        n_num.append(col)
        print()
        print(f"{adult_Df[col].value_counts()}")
print(f'no. of NUM Variables: {len(n_num)}')

## 2. CONTINUOUS VARIABLES
print()
n_cont = []
for col in adult_Df.columns:
    if adult_Df[col].dtype != object and len(adult_Df[col].unique()) >= 25:
        n_cont.append(col)
        print()
        print(f"{col} : Min_{adult_Df[col].min()} and Max_{adult_Df[col].max()}")
print(f'no. of CONT Variables: {len(n_cont)}')
print(f"Skipped: {abs((len(n_cat) + len(n_num) + len(n_cont)) - len(adult_Df.columns))} columns")


## Identifying Relationships
# 1. CrossTabs and PivotTables
## 1. CTabs workclass and EduClass -Two CAT vars.
workEdu = pd.crosstab(adult_Df['workclass'],adult_Df['EduClass'], normalize = True)
print(workEdu)

## 2. PTable HPW by Race and Salary
hrs_salRace = adult_Df.pivot_table(['hours-per-week'], ['salary', 'race'])
print(hrs_salRace)
# Amer-Indian-Eskimo work the longest for <= 50k
# Whites work the longest for > 50k, Blacks work the least hours in both cases.
## we can do the same for 'workclass', and so on.

# 2. Correlations
adult_corr = adult_Df.corr()
print(adult_corr.abs())

mask = np.zeros_like(adult_corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(20, 10))

sns.heatmap(adult_corr, mask = mask, center = 0, square=False, vmin= -1.2, vmax = 1.2,
            cbar_kws={"shrink": .5}, cmap="coolwarm", linewidths=.1, annot = True)
plt.show()
## The correlation here is not quite interesting and exempts a lot of non-numerical variables.

## Moving forward a method that strips CAT to numerical would be suitable,
# adult life is mostly related to independence it would be very applicable to thus,
# explore metrics like occupation, salary as dependent variables on these host of factors,
# to measure comfort or satisfaction.