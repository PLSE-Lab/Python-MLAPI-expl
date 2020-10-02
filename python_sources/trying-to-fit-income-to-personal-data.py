import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# Initialise variables
sample_size = 500000
column_dict = {'PWGTP': 'Weight',
               'DDRS': 'Self-care difficulty',
               'DEAR': 'Hearing difficulty',
               'DEYE': 'Vision difficulty',
               'SEX': 'Sex',
               'AGEP': 'Age',
               'SCHL': 'School Attainment',
               'ESR': 'Employment status'}
outcome = 'PINCP'  # Total person's income
predictors = list(column_dict.keys())
columns = list(predictors) + [outcome]
filename = '../input/pums/ss13pusa.csv'
row_count = 1613673

print("Prediction variables:")
for key, value in column_dict.items():
    print('  {}: {}'.format(key, value))

print("\nOutcome variable:\n  PINCP: Total person's income\n")

# Creating a random set of rows to skip from the total
np.random.seed(123)
skip_lines = np.random.choice(np.arange(1, row_count + 1),
                              (row_count - sample_size),
                              replace=False)

# Reading in the data, discarding other columns
df = pd.read_csv(filename, skiprows=skip_lines)
df = df[columns]

# Excluding data for children
df = df[df['AGEP'] > 15]

# Adding a column of ones to fit the intercept
df['const'] = np.ones((len(df), ))
predictors += ['const']

# Standard ordinary least squares regression
X = df[predictors]
Y = df[outcome]
result = sm.OLS(Y, X).fit()
print("Running linear regression (OLS)...\n")
print("T-statistic values:")
print(result.tvalues)

print("\nDiscarding lowest t-statistic \'DEYE\':")
predictors.remove('DEYE')
X = df[predictors]
result = sm.OLS(Y, X).fit()
print("T-statistic values:\n")
print(result.tvalues)

print("\nDiscarding lowest t-statistic \'DDRS\':")
predictors.remove('DDRS')
X = df[predictors]
result = sm.OLS(Y, X).fit()
print("T-statistic values:")
print(result.tvalues)

print("\nDiscarding lowest t-statistic \'PWGTP\':")
predictors.remove('PWGTP')
X = df[predictors]
result = sm.OLS(Y, X).fit()
print("T-statistic values:")
print(result.tvalues)

print("\nDiscarding lowest t-statistic \'DEAR\':")
predictors.remove('DEAR')
X = df[predictors]
result = sm.OLS(Y, X).fit()
print("T-statistic values:")
print(result.tvalues)

print("\nAll t values now large, plotting the data...")

# Plot the data by looping through remaining predictors
# and scatter plotting a selection of the data plus a
# fitted regression line
predictors.remove('const')
df = df.sample(50)
f, axarr = plt.subplots(1, len(predictors), figsize=(12.0, 5.0))
for i, predictor in enumerate(predictors):
    X = df[[predictor, 'const']]
    Y = df[outcome]
    result = sm.OLS(Y, X).fit()
    axarr[i].plot(X[predictor], result.fittedvalues, 'r-')

    axarr[i].scatter(X[predictor], Y,
                     marker='o',
                     edgecolor='b',
                     facecolor='none',
                     alpha=0.5)
    axarr[i].set_xlabel(column_dict[predictor])

    if i == 0:
        axarr[i].set_ylabel('Wages / Salary Income')
    else:
        plt.setp(axarr[i].get_yticklabels(), visible=False)

f.savefig("income_vs_personal_attr.png")
