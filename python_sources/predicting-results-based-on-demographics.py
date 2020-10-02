import numpy as np
import pandas as pd
demographicData = pd.read_csv("../input/county_facts.csv")
resultData = pd.read_csv("../input/primary_results.csv")
resultDataParty = resultData[resultData.party == "Republican"].reset_index()
resultDataGrouped = resultDataParty.groupby(["state_abbreviation", "county"])
winner = resultDataParty.loc[resultDataGrouped['fraction_votes'].transform('idxmax'),'candidate'].reset_index()
resultDataParty["winner"] = winner['candidate']
resultDataParty["totalVotes"] = resultDataParty["votes"]
votes = resultDataGrouped.agg({"votes": max, "fraction_votes": max, "winner": "first", "totalVotes": sum})
availableStates = resultData.state_abbreviation.unique()
#availableStates = ['IA', 'NV']
#availableStates = ['SC', 'NH']
availableStatesDemoData = demographicData[demographicData.state_abbreviation.isin(availableStates)]\
                                [['state_abbreviation', 'area_name', 'INC110213', 'RHI725214', 'RHI825214', 'EDU685213',\
                                  'SEX255214','SBO015207','PST045214','POP645213','POP815213']].reset_index()
availableStatesDemoData.rename(columns={'area_name':'county', 'INC110213':'income', 'RHI725214':'hispanic', 
                                'RHI825214':'white', 'EDU685213':'education', 'SEX255214':'females',\
                                'SBO015207':'femaleFirmOwner', 'PST045214':'population','POP815213':'nonEn_language',\
                                'POP645213':'notBornInUS'}, inplace=True)
availableStatesDemoData['county'] = availableStatesDemoData['county'].str.replace(' County', '')
del availableStatesDemoData['index']
availableStatesDemoData["income"] = availableStatesDemoData["income"]/1000
availableStatesDemoData = availableStatesDemoData.set_index(["state_abbreviation", "county"])
allData = pd.merge(votes, availableStatesDemoData, how="inner", left_index=True, right_index=True)
allData["turnout"] = allData.totalVotes/allData.population
sns.pairplot(allData, hue="winner", 
             x_vars = ["income", "hispanic", "white", "education", 'females'], 
             y_vars = ["fraction_votes"])
markerSize = (0.01+(allData.fraction_votes - allData.fraction_votes.min())/\
              (allData.fraction_votes.max() - allData.fraction_votes.min()))*300
g = sns.lmplot(x="white", y="income", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, size=6,
              legend_out=True)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("white percentage", size = 30,color="r",alpha=0.5)
g.set_ylabels("income (k$)", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 100), ylim=(22, 80))
g = sns.lmplot(x="white", y="hispanic", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, size=6,
              legend_out=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("white percentage", size = 30,color="r",alpha=0.5)
g.set_ylabels("hispanic percentage", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 100), ylim=(0, 32))
g = sns.lmplot(x="females", y="femaleFirmOwner", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("female (%)", size = 30,color="r",alpha=0.5)
g.set_ylabels("female firm owner (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(40, 55), ylim=(-2, 37))
g = sns.lmplot(x="white", y="turnout", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True, fit_reg=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("white (%)", size = 30,color="r",alpha=0.5)
g.set_ylabels("turnout (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 100), ylim=(-0.01, 0.27))
g = sns.lmplot(x="income", y="turnout", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True, fit_reg=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("income (k$)", size = 30,color="r",alpha=0.5)
g.set_ylabels("turnout (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 80), ylim=(-0.01, 0.27))
g = sns.lmplot(x="notBornInUS", y="nonEn_language", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True, fit_reg=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("not Born In US (%)", size = 30,color="r",alpha=0.5)
g.set_ylabels("persons not speaking English (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(0, 25), ylim=(0, 35))