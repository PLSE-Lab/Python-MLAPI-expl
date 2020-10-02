# %% [markdown]
# # Problem statement:
# Suppose that we have purchased bonds of 'Apple' worth 200000 dollars and have given a committment to the company worth another 200000. We need to calculate the expected loss of this portfolio now. 
# 
# Summary:
# 
# Outstanding(Bond) = 200000 dollars <br/>
# Committment(Future) = 200000 dollars

# %% [markdown]
# ### Some background theory
# #### We know the following equation for Expected loss of a portfolio :
# Expected loss(EL) = Exposure at default(EAD) x Probability of default(PD) x Loss given default(LGD)
# 
# Hence we need to find each of the three elements EAD, PD and LGD using different models in order to calculated our EL

# %% [markdown]
# ## PREPROCESSING: First of all, let us fetch the balance sheet data from the API 

# %% [code]
# Import libraries and build the API url
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import json

key = 'enter your API key here'

url = 'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/AAPL?period=quarter&datatype=csv&apikey={}'.format(key)

url

# %% [code]
# Function to fetch the data as a JSON file
def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

url = (url)
data = get_jsonparsed_data(url)

# %% [code]
# Check the JSON file structure
import pandas as pd
df = pd.DataFrame(columns = list(data['financials'][0]))
data['financials'][1]['date']

# %% [code]
data['financials'][0]

# %% [code]
# Build the balance sheet data as a dataframe
df['date'] = [data['financials'][i]['date'] for i in range(len(data['financials']))]
df['Total current assets'] = [data['financials'][i]['Total current assets'] for i in range(len(data['financials']))]
df['Total liabilities'] = [data['financials'][i]['Total liabilities'] for i in range(len(data['financials']))]
df['Total non-current assets'] = [data['financials'][i]['Total non-current assets'] for i in range(len(data['financials']))]                                                       
df['Total assets'] = [data['financials'][i]['Total assets'] for i in range(len(data['financials']))]
df['Total current liabilities'] = [data['financials'][i]['Total current liabilities'] for i in range(len(data['financials']))]
df['Total liabilities'] = [data['financials'][i]['Total liabilities'] for i in range(len(data['financials']))]
df['Total current liabilities'] = [data['financials'][i]['Total current liabilities'] for i in range(len(data['financials']))]
df['Total non-current liabilities'] = [data['financials'][i]['Total non-current liabilities'] for i in range(len(data['financials']))]
                                                        

# %% [code]
# Save the dataframe as a csv file for further use
df = df.dropna(axis=1)


# %% [markdown]
# ## EAD MODEL: Internal credit risk model by Michael Ong

# %% [code]
df.head()

# %% [markdown]
# #### Exposure at default comprises of two non risk-averse parts:
# 1. Outstanding (Bonds)
# 2. Committment x Usage given default
# 
# Hence we would need to first calculate the Usage given default(UGD)

# %% [markdown]
# Upon checking Apple's website, we find that the credit rating for the company is AA+ which accounts for 73% of UGD

# %% [code]
# Let us assign the necessary variables
bond = 200000
comt = 200000
ugd = 0.73

ead = bond + (ugd*comt)
print("The exposure at default(EAD) is:", ead)

# %% [markdown]
# ## PD MODEL: Merton Structural model

# %% [code]
# Let us create a new column which we can use for plotting
import numpy as np
df['quarterspan'] = [i/4+0.25 for i in range(len(df))]
df['quarterspan'] = df[['quarterspan']].shift(1)
df['quarterspan'] = df['quarterspan'].replace(np.NaN,0)
df.drop('date', inplace = True, axis = 1)
df = df.astype(np.float64)

# %% [code]
# Let us plot the assets and liabilities per quarter to see the trend
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
fig = plt.figure()

ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)
ax0.set_title("Asset value plotted with quarters")
ax1.set_title("Liabilities plotted with quarters")

df.plot(kind = 'line', x = 'quarterspan', y = ['Total assets','Total current assets','Total non-current assets'], ax=ax0, figsize = (17,5))
df.plot(kind = 'line', x = 'quarterspan', y = ['Total liabilities','Total current liabilities','Total non-current liabilities'],figsize = (17,5),ax=ax1)
plt.show()

# %% [markdown]
# We will now proceed to assign and calculate the values which will be used in the merton structural model calculations

# %% [code]
# Calculating some essential measures for the model
firm_value = df[['Total assets']].iloc[0,0]
rate = 6.25/100
time = 1
volatility =  df['Total assets'].std(ddof=1) / df['Total assets'].mean()
print("The firm value is {} and the rate of interest is {}% and the volatility according \
to asset levels is {}%".format(firm_value, rate*100, round(volatility * 100, 2)))

# %% [code]
stl = max(df['Total current liabilities'])
ltl = max(df['Total non-current liabilities'])
default_pt = stl + (0.5*ltl)
print("If the company's total assets hit {} , it will default".format(default_pt))

# %% [code]
df.columns

# %% [markdown]
# Now we will calculate the distance to default

# %% [code]
import math
numerator = math.log(firm_value/default_pt) + ((rate - np.power(volatility,2)/2)*time)
denominator = volatility*time
dd = numerator/denominator
from scipy.stats import norm
PD = norm.cdf(-dd)
print("The probability of default(PD) as per the Merton structural model is {}%".format(round(100* PD,2)))

# %% [markdown]
# ## LGD MODEL: Gamma distribution

# %% [markdown]
# According to Moody's ultimate recovery database, for bonds the LGD mean is around 56.37% and the standard error is 1.04%
# 
# We also learned from an article from Moody's that the maximum value of the distribution for a bond is set to 1.1

# %% [code]
# Calculating the alpha and beta values of the beta distribution 
mu = 0.5637
dev = 0.0104
maxm = 1.1

alpha = mu/maxm * (mu*(maxm - mu)/(maxm * np.power(dev,2))-1)
betaa = alpha*(maxm/mu-1)
print("The alpha is {} and the beta value is {}".format(round(alpha,2),round(betaa,2)))

# %% [code]
# Calculating the LGD ( LGD = 1 -mean_recovery)
mean_recovery = alpha/(alpha+betaa)
LGD = 1-mean_recovery
print("The loss given default(LGD) is {}%".format(round(100*LGD,2)))

# %% [code]
# Visualizing the beta distribution
from scipy.stats import beta
me, var, skew, kurt = beta.stats(alpha, betaa, moments = 'mvsk')

x = np.linspace(beta.ppf(0.01, alpha, betaa), 
               beta.ppf(0.99, alpha, betaa), 100)
plt.plot(x, beta.pdf(x, alpha, betaa),
       'r-', lw=5, alpha=0.6, label='beta pdf')
plt.title("The constructed beta distribution")
plt.show()

# %% [markdown]
# ## CALCULATING EXPECTED LOSS

# %% [markdown]
# Expected loss(EL) = EAD x PD x LGD

# %% [code]
print("The expected loss of our portfolio is ${}".format(round(ead*PD*LGD,2)))