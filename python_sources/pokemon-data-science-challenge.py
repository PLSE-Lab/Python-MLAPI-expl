#!/usr/bin/env python
# coding: utf-8

# # Pokemon Data Science Challenge

# ![](https://lh3.googleusercontent.com/nuqQn4oUVnmqxe0pq8EcofuXLNIZ9gg6JkxFuvukxnClQ7Zy2eWA1Y1oBeYqlYJ3tZ29pHBsXQ6rHh03Tji9gxB1OeWGfsD0oy0Q48oJ_LMAFuq9HZqkh_hWQiSmKlJuGZZHSXrDRjkh43rKeJmDlCE9b6MeX6FMHuay3aniKPVr6pSEXltIS-98RnCjqQMtD6mi7ddtDvG4aSVIXI-D227U192AKaRsCQpv0pi9hUxu5j1-kyaV_uI4LuKkjosf9M1piA85CbxLr-XgFT6BGz1PrxEsOFzAXM7t0ja6jBg399X6DXmVappJSOcPkGoeIlrZqTn50WfCtNrL7ACdVsqVavoNbL74xjPki7HAAmTwmEk7Vkt068O2Hnth4V3l4pqZlRU9miN-v1s2EjydRKv98jnqkiXJuKUdrxK-nBAeZdhymYcU5bcYIiGHbxXZOq3Dzmi77X_PeWEv-wCxKz7Mz3NP2dwBVRC64LwNVPQFhQO0S-iqG_aLUTdppclphJlUe_4gJ0lSYrQ26KFqGS8sAyfeIKexOjzGRvuiRu2H_A8u_RbQAjCFc2oSH-fBRxjFe7am8y_Q8T1cxamQ5YdY7UYFHRnLK1cQXZf5TQ=w2048-h1193-no)

# I'm Kaggle Beginner. This is my first Kaggle Kernel to learn data analyzing method. 

# ### Contents
# #### 1. Data Analysis & Visuallization
# 1-1. Simple Analysis  
# 1-2. Analysis on Base stats  
# 1-3. Why it happen superior pokemons lose to inferior pokemons?        
# 1-4. What pokemon have most high percentage of winning?
# 
# #### 2. Data Cleansing & Predicting
# 2-1. Completing Data  
# 2-2. Preparing DataFrame for predicting   
# 2-3. Regression, Classifier and Verification  
# 2-4. Predicting  

# #### 1. Data Analysis & Visuallization
# ##### 1-1. Simple Analysis

# In[ ]:


import pandas as pd
combats = pd.read_csv('../input/combats.csv')
combats.head(3)


# First row is No.266 pokemon vs No.298 pokemon battle, and No.298 pokemon is Winner.
# What are No.266 pokemon and No.298 pokemon? Check on pokemon.csv.

# In[ ]:


pokemon = pd.read_csv('../input/pokemon.csv')
pokemon_266_298 = pokemon[pokemon['#'].isin([266, 298])]
pokemon_266_298


# No.266 pokemon is Lavitar. No.298 pokemon is Nuzleaf.

# <img src="https://lh3.googleusercontent.com/TECjRWeu5CdKIgHDgVsV5VI7r0_EARgGEFPxTUo8sR9Cu0yRtzo3iAYDpv9Q_9vx9_DtKiJo-2rTKA5vDPoRGhAg2zP3gmdaMO9gIv_iEvECIhv8pNzSjD7kxM8W7jM92XuAUDS5ix_mTUvksr5ZUf6Wet7Nu_xcYG31OHqGCg51iS9U9boI6h6xv2-ItbKUrBFBanoBIvoUWcvGfI1KV3RlgmAYkWo5RgSjLC8PE0DnSIs6IvDNAebajBbok8mHT_BEgV8yFfpZ3Dsrk8T2mPeOmQDOiQvp6mnF6JegAbhAwE-ZKBIFn-FO5rTHnDWymC4M7Ur1jSVRv2P06sYtJR5ZXbb_OmvEFM5VzgsFnkSAFnovYtMyENcowhNop5QYt9x22Y6sO_tsjYu2YonyA5bV-NoKcUHEHzkCFxcSLuRsTSWYPeENTOnsPdXZAbGG2MnGulxgRcnTJbln8KkCqWaiw4Yz8q8l3muD98HYCCpodSeRTazgW6p4t6HYqAVyDd0qi2rODFPxTEzeuo6ECQqA0ZLIlgYn8zhj0NAsj8zOa4yrrnyJ1Jwfneq2vmCkTP4zvFpTFK9ztQw0dPiOFzF2lkZMq8UGmj0hzaBGDA=w302-h153-no">

# Winner is Nuzleaf. Nuzleaf's type is Rock and Ground, and Lavitar's type is Grass and Dark. Pokemon Type Chart is a following figure. According to this type chart, Nuzleaf has type advantage. 

# <img src="https://lh3.googleusercontent.com/TZvRRKFXFCy-hocOUCvojB0jdf3OUn2yqn8AB_3QfSGo3iIxfJY2eLO0iwC6zPxudmDlIlyBigeyb5QpzLW6-s9vqyMxEFIhIwJPfR5NIBCnXZsCBejL2Fp6YrqZt-mBjY6b8MTErkIFadk8pc7BN-ea57V2b0Ae3EZLm4S9VZyqp7QhPFjBNBr0k6QQShNqotZVSeDnK79uJ68V35nVXHU4WXcAi3iaV5yoMwGbgO6vZw2a3rhOCtuPRNG2sCvQvAgtt1yWMCrXsbAqLqMVXOpWzq2h_7yHUJEKbaLM15I2agIQvsLCL6982ukzf-Oc6RMlxt5SYGb5nBfklScJ0hcL4CNa10MREurHCaISNWObzmI5TBzRpW-_tawpezkVn7a3qCqJpMN70USqnjn4sqBhToOIZOxmLUXJCVBBPEB7jMD3OHHLibXOQ2vaK4JGxsKZxKmN5zbzyHGBFeg-nyL2eTQwPDb2msxoOgma0w_teKUL3V2LvkQuRRYDL3Qv_RySk5bDhTN9l_6odXdGXQHB4G-buYhOgRnUktGme8cSn2BpSk4Q7wac4PeNIsJexUmDu93ggQdggOMeHEFkvpa9kheLHriW_DnvZrAzJw=w1160-h571-no" width="80%">

# In[ ]:


names_dict = dict(zip(pokemon['#'], pokemon['Name']))
cols = ["First_pokemon","Second_pokemon","Winner"]
combats_name = combats[cols].replace(names_dict)
combats_name.head(3)


# <img src="https://lh3.googleusercontent.com/1S3uqz2lvZrjcdNsC_cXc6oVZWPhgW-Uv3sEWoZgu6_DH9c7uEmJ1I8lgebCVJvWbkIHqp7F5TU_Bi745vh8C6Lj-KWWw0LaJX55JooE-UYWXJpGDnJyz9hDnHrUJVUaH8Ug4XEYDMGEVAnF8LYtHnZ5n59JRnSh4YLci8AKGCRQ_o3ZOU1-J86C0N4pZ2zJF7JVrm68_ivxa045BbY5aPXqb_N88UeG0kwIYOFTn9LRTR5hHsf-TT4vGvcSTKeKKKpmR1gHfMX2p5DGbcqnpX7EnNCGzuqWIDOz-jZTFXNNmyxx7QzP6f2c2a9-DBpOVVWkvrU72meKhqlwR5JwgtAZTrtdXupqz1_E_oeaWpmJW5yLJ09wSIw-zM-FOE408hvJ6QMYR5nIaARW0rH7uPh46Dal-mfRQYUXCLMofX2IdKUqdb81WIGg7ZTXmXsGttnRWzzGPfKA5wnvsToq1AEkrunJwXqy1OpGZ_EMWHtj1c73MRokTcYNxNCEUez2Y6yL5Zdkic4P3vaZ5VN36FMnZF_KPxV6gZHKcscCLrEpekEBZpR518mSW4OLK26Uq_yrg6wOsefitO5Mpv22PYZEKhz60709PxPGAnQShw=w302-h452-no">

# In combats file, Head 3 Battles and Winners: 
# 1. Larvitar (Rock/Ground) vs Nuzleaf (Grass/Dark)         -> Winner: Nuzleaf
# 2. Virizion (Fighting/Grass) vs Terrakion (Fighting/Rock) -> Winner: Terrakion
# 3. Togetic (Fairly/Flying) vs Beheeyem (Psychic)          -> Winner: Beheeyem

# ##### 1-2. Analysis on Base stats
# In Pokemon Game, species's base stats are an important defining characteristic. Pokemons with high base stat total are strong and popular on battle. Check the base stats total distribution for all pokemons.  

# In[ ]:


pokemon["stats_sum"] = pokemon["HP"] + pokemon["Attack"] + pokemon["Defense"] + pokemon["Sp. Atk"] + pokemon["Sp. Def"] + pokemon["Speed"]
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
print(pokemon["stats_sum"].describe())
sns.distplot(pokemon["stats_sum"])
plt.show()


# This distribution histogram shows that base stat total's distribution has a bimodal distribution. First modal is around 300 and the other is around 500.
# Next, Check the difference distribution of total base stats between one pokemon and the other pokemon in each match. 

# In[ ]:


stats_sum_dict = dict(zip(pokemon['#'], pokemon['stats_sum']))
combats_stats_sum = combats[cols].replace(stats_sum_dict)
diff_stats_sum = abs(combats_stats_sum["First_pokemon"]-combats_stats_sum["Second_pokemon"])
print(diff_stats_sum.describe())
sns.distplot(diff_stats_sum)
plt.xlabel("diff_stats_sum")
plt.show()


# On this distribution, median is 118. Around 50% match are the battle has the difference more than 100 values.
# Then, did pokemon with higher total base stats win a battle against pokemon with lower total base stats?

# In[ ]:


combats_stats_sum["Loser"] = combats_stats_sum.apply(lambda x: x["First_pokemon"] if x["First_pokemon"] !=  x["Winner"] else x["Second_pokemon"], axis = 1)
diff_win_lose_stats = combats_stats_sum["Winner"] - combats_stats_sum["Loser"]
print(diff_win_lose_stats.describe())
sns.distplot(diff_win_lose_stats)
plt.xlabel("diff_win_lose_stats")
plt.show()


# This distribution shows there is a tendency that pokemon with higher total base stats win a battle against pokemon with lower total base stats. It looks like "A base stats total is a main factor for winning". On the other hand, there are battles pokemon with lower base stats surpasses one's superiors.
# Check the combination battles pokemon with lower base stats surpasses one's superiors with more than 100 values.

# In[ ]:


stats_sum_dict_re = dict(zip(pokemon['stats_sum'], pokemon['#']))
combats_stats_sum["diff"] = diff_win_lose_stats
surpassing_stats_sum = combats_stats_sum[combats_stats_sum["diff"] < -100]
print ("Surpassing one's superiors Battle number : " + str(len(surpassing_stats_sum)))
surpassing_id = surpassing_stats_sum[cols].replace(stats_sum_dict_re)
surpassing_name = surpassing_id[cols].replace(names_dict)
surpassing_name.join(combats_stats_sum["diff"]).head(8)


# The number of combination has the difference more than 100 total values is 5716 combinations. These combinations account for 10% of all battles. Head 8 combinations and winners are below.

# <img src="https://lh3.googleusercontent.com/TqGv0RVdharxeKsZuq9sSqnV0l20b848jnLO2jKpKpHw_8Txi7_LaEJNpk_r2GOBfJqtv1rHHje2VL_99d8ZoyWLJnuPFw2CKKfRnI96l9CaqcCYDXKE1w8zf5rIF3nd5NMg8gZeo_RgLsA7_XOpnAG7PI3jE8fqsnhYqlpSs5iGa_YolUxqysmjSKQ8Gcw4GuQdBWTYWi7kqC43zs9wCx8A0EsqJJD2WJIfQnQkZfLyt49go8N-v47LX920ONSTiAqMBO2b5OL4WyROPffawR_TsD2PdpNcCDmuhoLVC49Ysz4VU46ctxmoc0wFni1i1CkXT8hC47MqNr3oOIG5sgS4qNI5UdegCzXlLOBaOgy1ZA7roUq2lwCnDu5cUlqljxJP0mtrjUO7ME-9g3e-2phQit484s7UNdP7H_NOXw5rxKTTot-i7_d3veuoL4ih0aqh8mwD-QgOjX9I7FSgZz_l0h2sd-o2nrSl962mDs-LOzsuewwpbiOvpMglLOYEUN-gvv0-cpE2X1uAj-vkb944WMCaR8oDFqWb2_ENLff9dICOLpgJ13pmM_ffTayYKhED8AyaJrl_28Sn2HB2-b9VbSZ8yHPnar3qSRrcng=w555-h540-no">

# I want to know the combination has max difference of total base stats. 

# In[ ]:


surpassing_name.join(combats_stats_sum["diff"]).sort_values(by="diff").head(4)


# <img src="https://lh3.googleusercontent.com/x4rmtbvelaN-zr8vUZx9Nh863m_ze6lB0HJ6Ywi_qsKy7HwUgJbESIikmbNsvDdldJpy3lNpH-qq8CaSbQSFREwVyCwRy36mq6OOuZGK4dRYYkXjlenOHBQTB2Q6QJTe8ruV9ZS_lu2ur_L5f65DzxeP1mJXNQG5bps1QURFTsAV7LOHfPrya-afE7bo8FI23iy7gloo7VKDhlyGqcoyvSxLpQsKePj5SL0K5lIowqBgfW4WtWVUm3k7f_nPBfpa14Upby1zKHMP00rMvU_SKfjimmj525DjNHXJ6j-F2GCH1yD19jskk1UnSFOaUJn9y_9cUNnFSKMuUcHmLjPXaQVEJAThugEd_pGwCvgdhit3n7F8O6QkGoemVNkNsyRLFUtL9-QtjCWIz-huzung9ibGDHAjO2M5MKUXFhufgf1VFT7uONROu6W-1FdxvYUe8Jbi_vwth9EIQK3oJkbuLI2ea5DK0OetGtLyMCD8KK6VnxLdeKT6w5C3SjBUXcw4fwWCxucEffw0ukYS3P-_4ePxBCkhvYXPX8Lo56EuMI1Ggf9ga5ZY5EKtpsP08cK2ZrxQ8puuOTpGzM4Lyy-JAG9IijAfz_9pgOgvFB98KQ=w255-h381-no">

# Mega Rayquaza has lost to pokemons with much lower base stats. The difference between Mega Rayquaza and Clefa is 562. 

# ##### 1-3. Why it happen superior pokemons lose to inferior pokemons?
# In my idea, there may be 2 factors.
# 1. First Attack
# 2. Type advantage
# 
# Firstly, Check the effectiveness of first attack. 

# In[ ]:


combats_stats_sum["First_Win"] =  combats_stats_sum.apply(lambda x: 1 if x["First_pokemon"] ==  x["Winner"] else 0, axis = 1)
surpassing_stats_sum = combats_stats_sum[combats_stats_sum["diff"] < -100]
sns.distplot(surpassing_stats_sum[surpassing_stats_sum["First_Win"]==0]["diff"], label="First_pokemon_win=0")
sns.distplot(surpassing_stats_sum[surpassing_stats_sum["First_Win"]==1]["diff"], label="First_pokemon_win=1")
plt.legend()
plt.show()


# No difference between First attacker win and lose based on the base stats difference.  
# Is it same on all combination battles?

# In[ ]:


sns.distplot(combats_stats_sum[combats_stats_sum["First_Win"]==0]["diff"], label="First_Win=0")
sns.distplot(combats_stats_sum[combats_stats_sum["First_Win"]==1]["diff"], label="First_Win=1")
print ("-First_Win=0-")
print (combats_stats_sum[combats_stats_sum["First_Win"]==0]["diff"].describe())
print ("-First_Win=1-")
print (combats_stats_sum[combats_stats_sum["First_Win"]==1]["diff"].describe())
plt.legend()
plt.show()


# No difference on all battles. It's actually that the count of battle which a first attacker win is less than one which a first attacker lose. So First Attack is not important on result.

# Next, Check the effectiveness of type advantage. How many types are there in pokemon world?

# In[ ]:


print ("There are {} Types.".format(len(pokemon["Type 1"].drop_duplicates())))
list(pokemon["Type 1"].drop_duplicates())


# Now, Pokemons can have two types. How many type combinations are there?

# In[ ]:


type_cols = ["Type 1", "Type 2"]
print ("There are {} unique type-combinations.".format(len(pokemon[type_cols].drop_duplicates())))


# In[ ]:


pokemon["Type 2"] = pokemon["Type 2"].fillna("None")
type_cross = pd.crosstab(pokemon["Type 1"], pokemon["Type 2"])
type_cross.plot.bar(stacked=True, figsize=(14,4))
plt.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', ncol=5, fontsize=8, title="Type 2")
plt.show()


# Let's make type chart dataframe.
# If the type of an attacker is super effective against a type of its target, the damage done is double the normal amount;
# If the type of an attacker is not very effective against a type of its target, the damage done is half the normal amount;
# If the type of an attacker is not effective against a type of its target, the target is completely immune to it, and the move will deal no damage.

# In[ ]:


Normal = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 1, "Rock": 0.5, "Ghost": 0, "Steel": 0.5, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 1, "Psychic": 1, "Dragon": 1, "Dark": 1, "Fairy": 1}
Fighting = {"Normal": 2, "Fighting": 1, "Poison": 0.5, "Ground": 1, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0, "Steel": 2, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 2, "Psychic": 0.5, "Dragon": 1, "Dark": 2, "Fairy": 0.5}
Poison = {"Normal": 1, "Fighting": 1, "Poison": 0.5, "Ground": 0.5, "Flying": 1, "Bug": 1, "Rock": 0.5, "Ghost": 0.5, "Steel": 0, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 2, "Ice": 1, "Psychic": 1, "Dragon": 1, "Dark": 1, "Fairy": 2}
Ground = {"Normal": 1, "Fighting": 1, "Poison": 2, "Ground": 1, "Flying": 0, "Bug": 0.5, "Rock": 2, "Ghost": 1, "Steel": 2, "Fire": 2, "Water": 1, "Electric": 2, "Grass": 0.5, "Ice": 1, "Psychic": 1, "Dragon": 1, "Dark": 1, "Fairy": 1}
Flying = {"Normal": 1, "Fighting": 2, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 2, "Rock": 0.5, "Ghost": 1, "Steel": 0.5, "Fire": 1, "Water": 1, "Electric": 0.5, "Grass": 2, "Ice": 1, "Psychic": 1, "Dragon": 1, "Dark": 1, "Fairy": 1}
Bug = {"Normal": 1, "Fighting": 0.5, "Poison": 0.5, "Ground": 1, "Flying": 0.5, "Bug": 1, "Rock": 1, "Ghost": 0.5, "Steel": 0.5, "Fire": 0.5, "Water": 1, "Electric": 1, "Grass": 2, "Ice": 1, "Psychic": 2, "Dragon": 1, "Dark": 2, "Fairy": 0.5}
Rock = {"Normal": 1, "Fighting": 0.5, "Poison": 1, "Ground": 0.5, "Flying": 2, "Bug": 2, "Rock": 1, "Ghost": 1, "Steel": 0.5, "Fire": 2, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 2, "Psychic": 1, "Dragon": 1, "Dark": 1, "Fairy": 1}
Ghost = {"Normal": 0, "Fighting": 1, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 1, "Rock": 1, "Ghost": 2, "Steel": 1, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 1, "Psychic": 2, "Dragon": 1, "Dark": 0.5, "Fairy": 1}
Steel = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 1, "Rock": 2, "Ghost": 1, "Steel": 0.5, "Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Grass": 1, "Ice": 2, "Psychic": 1, "Dragon": 1, "Dark": 1, "Fairy": 0.5}
Fire = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 2, "Rock": 0.5, "Ghost": 1, "Steel": 2, "Fire": 0.5, "Water": 0.5, "Electric": 1, "Grass": 2, "Ice": 2, "Psychic": 1, "Dragon": 0.5, "Dark": 1, "Fairy": 1}
Water = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 2, "Flying": 1, "Bug": 1, "Rock": 2, "Ghost": 1, "Steel": 1, "Fire": 2, "Water": 0.5, "Electric": 1, "Grass": 0.5, "Ice": 1, "Psychic": 1, "Dragon": 0.5, "Dark": 1, "Fairy": 1}
Electric = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 0, "Flying": 2, "Bug": 1, "Rock": 1, "Ghost": 1, "Steel": 1, "Fire": 1, "Water": 2, "Electric": 0.5, "Grass": 0.5, "Ice": 1, "Psychic": 1, "Dragon": 0.5, "Dark": 1, "Fairy": 1}
Grass = {"Normal": 1, "Fighting": 1, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 1, "Steel": 0.5, "Fire": 0.5, "Water": 2, "Electric": 1, "Grass": 0.5, "Ice": 1, "Psychic": 1, "Dragon": 0.5, "Dark": 1, "Fairy": 1}
Ice = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 2, "Flying": 2, "Bug": 1, "Rock": 1, "Ghost": 1, "Steel": 0.5, "Fire": 0.5, "Water": 0.5, "Electric": 1, "Grass": 2, "Ice": 0.5, "Psychic": 1, "Dragon": 2, "Dark": 1, "Fairy": 1}
Psychic = {"Normal": 1, "Fighting": 1, "Poison": 2, "Ground": 2, "Flying": 1, "Bug": 1, "Rock": 1, "Ghost": 1, "Steel": 0.5, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 1, "Psychic": 0.5, "Dragon": 1, "Dark": 0, "Fairy": 1}
Dragon = {"Normal": 1, "Fighting": 1, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 1, "Rock": 1, "Ghost": 1, "Steel": 0.5, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 1, "Psychic": 1, "Dragon": 2, "Dark": 1, "Fairy": 0}
Dark = {"Normal": 1, "Fighting": 0.5, "Poison": 1, "Ground": 1, "Flying": 1, "Bug": 1, "Rock": 1, "Ghost": 2, "Steel": 1, "Fire": 1, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 1, "Psychic": 2, "Dragon": 1, "Dark": 0.5, "Fairy": 0.5}
Fairy = {"Normal": 1, "Fighting": 2, "Poison": 0.5, "Ground": 1, "Flying": 1, "Bug": 1, "Rock": 1, "Ghost": 1, "Steel": 0.5, "Fire": 0.5, "Water": 1, "Electric": 1, "Grass": 1, "Ice": 1, "Psychic": 1, "Dragon": 2, "Dark": 2, "Fairy": 1}

type_relation = {"Normal": Normal, "Fighting": Fighting, "Poison": Poison, "Ground": Ground, "Flying": Flying, "Bug": Bug, "Rock": Rock, "Ghost": Ghost, "Steel": Steel, "Fire": Fire, "Water": Water, "Electric": Electric, "Grass": Grass, "Ice": Ice, "Psychic": Psychic, "Dragon": Dragon, "Dark": Dark, "Fairy": Fairy}
df_type_relation = pd.DataFrame(type_relation)
print ("Row is Diffender, Column is Attacker")
df_type_relation


# Make "Type" column which shows type advantage as a numeric. "0" shows no damage will be changed to 0.25 not to become zero when it be multipled. 

# In[ ]:


pokemon["Type"] = pokemon.apply(lambda x: x["Type 1"]+"/"+x["Type 2"], axis=1)
type_dict = dict(zip(pokemon['#'], pokemon['Type']))
combats_type = combats[cols].replace(type_dict)
combats_type["Loser"] = combats_type.apply(lambda x: x["First_pokemon"] if x["First_pokemon"] !=  x["Winner"] else x["Second_pokemon"], axis = 1)

zero_dict = {0: 0.25}
df_type_relation = df_type_relation[:].replace(zero_dict)

def calcRelation(combats_type):
    r0 = 1
    win_type1 = combats_type["Winner"].split("/")[0]
    win_type2 = combats_type["Winner"].split("/")[1]
    lose_type1 = combats_type["Loser"].split("/")[0]
    lose_type2 = combats_type["Loser"].split("/")[1]
    if win_type2 != "None" and lose_type2 != "None":
        r1 = df_type_relation[win_type1][lose_type1]
        r2 = df_type_relation[win_type1][lose_type2]
        r3 = df_type_relation[win_type2][lose_type1]
        r4 = df_type_relation[win_type2][lose_type2]
        r = r0 * r1 * r2 * r3 * r4
    elif win_type2 != "None" and lose_type2 == "None":
        r1 = df_type_relation[win_type1][lose_type1]
        r3 = df_type_relation[win_type2][lose_type1]
        r = r0 * r1 * r3
    elif win_type2 == "None" and lose_type2 != "None":
        r1 = df_type_relation[win_type1][lose_type1]
        r2 = df_type_relation[win_type1][lose_type2]
        r = r0 * r1 * r2
    elif win_type2 == "None" and lose_type2 == "None":
        r1 = df_type_relation[win_type1][lose_type1]
        r = r0 * r1
    return r

combats_type["Relation"] = combats_type.apply(lambda x: calcRelation(x), axis = 1)
print (combats_type["Relation"].describe())
sns.distplot(combats_type["Relation"])
plt.show()


# The Distribution of Type advantage number doesn't have a very distinctive features.

# ##### 1-4. What pokemon have most high percentage of winning?
# Firstly, Check "What pokemon appears high frequency in "Winner" colummns?" by using wordcloud.

# In[ ]:


import numpy as np
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

combats_names = combats[cols].replace(names_dict)
print (combats_names["Winner"].value_counts()[:10])
winners = list(combats_names["Winner"])
winners_str = [str(i) for i in winners]
winners_text = (",").join(winners_str)
wc = WordCloud(background_color= "black", random_state=1, margin=3).generate(winners_text)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(wc, interpolation='bilinear')
plt.show()


# On WordCloud image, Therian Forme and Incarnate Forme are biggest characters. They are common forme to 3 pokemon "Tornadus", "Thundurus", and "Landorus".

# <img src="https://lh3.googleusercontent.com/gcJ-QWslS3cyTlEoOr4Gwd1xmXkiB_7m0bM7ZK6VItyAtcSPW7tGooMym-2ZRSXTFkTEFxl9TC5yBxTmiZKABH7P5pb3RvfBuIXdcwMZLCnRn6VXvtK_FePyRYJ48_UQHS5zynCWOjIkNNzwlJWxQMJaqvJOx1bamdd5IjqkmHW-du_RdV2yLe9AJzMdV563CHA87UeHfc30Tn0QCwlBQ9WN5kc_lx2aI8sY25bKaVmSuAEHjjffvUI1dLURJORhAsvcBy8oRegKJGhGslF_AImlFB3mSkwGZ86AcLCiptRheaaiOpjQVi5l0NjwgvAla-lp1tUbDA22IQLcXZugWd3dQFe9TsB7UXQJFJlDQu4nPknjUGhkBOcT72Dg9xkmAn_mMg6tWFRefX1y0MULL43M2oa0RU69_N1yWBctChuAx1T_iJAeiVNQigXZ9UTjg-5UT_6czNrnCtZTIStSTxmZ1ePNNkV1yK14nlAAdjzJ-_cQb5eRaoqM8mk10pxl0ADpMezL8WGiZ4MJSYY5kQqHlsm9Z_zujJR2kwHrwIbZUfjkuz3PyPlOQfK7RcDcyGL8moMhIHZkFyB7yNYt5LEnZMzPXvxlNhETBEMsLw=w657-h240-no">

# Best 10 winning pokemons are Mewtwo, Aerodactyl, Infernape, Jirachi, Deoxys Speed Forme, Slaking, Murkrow, Mega Absol, Mega Houndoom, Mega Aerodactyl.
# However, I want to know pokemons have high winning percentage.

# In[ ]:


first_num = combats_names["First_pokemon"].value_counts()
second_num = combats_names["Second_pokemon"].value_counts()
battle_num = first_num + second_num
battle_win = pd.DataFrame({"battle": battle_num, "win": combats_names["Winner"].value_counts()}, columns=["battle", "win"])
battle_win["ratio"] = battle_win["win"]/battle_win["battle"]
battle_win.sort_values(by=["ratio"], ascending=False).head(10)


# Best 10 winning percentage pokemons are Mega Aerodactyl, Weavile, Tornadus Therian Forme, Mega Beedrill, Aerodactyl, Mega Lopunny, Greninja, Meloetta Pirouette Forme, Mega Mewtwo Y, Mega Sharpedo.

# <img src="https://lh3.googleusercontent.com/H47JXVJoRxU2_jdPU8MPQHkKREVsNpgoU3e07EkYBnFvWaE9xTLW6NCcM7vjSW3Mpot2_8kv5eN0MTJVbGK0yZWPEkY_apbgN_LpMHCNE22kpED5GY8XaaDh0GJYwBkMK9VUHy6mVxxupVVtInCLcilX61kBiTkDEV984ScRxTCPxaatbdRlr9XndX5AM_BSOejC3_bj7RxaQAh_-NnDCDiLOUByJm_xrAJXjAqFEA4sYCaOIlroelQ2A1J3wn0avXqpe5EoHvL148wJliynOxyuK4e-A1XVDwirbGhxXXk6eBIfa2i_YAQVFFoDJfmwylbu41DiC7TkGMO3NA0nu1Ud7OSbReLei3rSDBQOafDzKRwfPzWGcQGRg8nRMFV_ypL6xOa5YMH8hapQqZ07RRFpjH04DqbSiZDdY2WHXIT27ns4mi4Ls8gRDyJpb4denSXmYuoAMYUGM6rffiUL3fnHgx8Ok4Jpu89bA0ELzNVZVnceaUlmUNPEOfGOzgm7mdA_ZeC95QVpxiDjY6OyPK0SOIJh5t0YZAOb-ITJARGEsVLKVbTfa7aU0DdSei-8xI9iIu5oL_I36I2irNjrz7VU2zmiRyo6tJBixxnNXQ=w720-h358-no">

# #### 2. Data Cleansing & Predicting
# 
# ##### 2-1. Completing Data

# In[ ]:


battle_win.info()


# On above information, there are 783 pokemons. It's not 800. 17 pokemons didn't experience a battle.
# And 1 pokemon of 783 pokemons doesn't have win and ration values.

# In[ ]:


battle_win[battle_win["win"].isnull()]


# <img src="https://lh3.googleusercontent.com/RMxE6M4avryHEEmBSIyQWAppHrRsxAJaiqRfUSqXsCQmV5aZd1t6RY_iLS27fSZKcjzhVfiFJX3od2lrtDzdrgqMilgXSCdqdeYRRi5je8-RYHyj1bLM-7HKrwtGicgpgS4tmPt0QQK5IDR2h_fmBkiM8Z2Pa91iVqFgbODYcSpPoIN6GpVrFZb5_6MuMOFHPdtrJBWVoKBkY1943JK8CKcXotepYS8wE_aDdt8QTxThL5T_3RoVxSajEtJaOcXmtovJiFrSEUrCiIw3aZ3hIhWyhQ1OfIsdsakPcM3WVJjfMt1_neaxE5OyxZUt0Ato6ChBggGrXRdUsgGlWou5SrThVyTFIflLl8Q7-xjUAHXVSqpyW93WaSbihaG96ejZehMoy8VDOYWkg4AQPugzWVY4DDv7cHDpxMAl2PzACA6HRxPVxmwypBwU-57CyKNqenZzWUo-ZUOTVk-0kyBvVj8i3f-Anqx0piVpDSf7NHEtqOYgTMUI4nCLDrB9cN9iWkVLrHji7tOX6QZ_8lhXjJSrOfVNUw75jvF-JUXc9pKUBbOkYSqYDyBOmN6LHjfKN9uRxV-qK-JUNZU1Q158FMvc9hsW_Tu8SMZDMid8MA=s500-no" width=30%>

# Shuckle has experienced 135 battles, but could not experience winning. Fill zero value to "win" and "ratio" column of Shuckle.

# In[ ]:


battle_win.loc["Shuckle", ["win", "ratio"]] = 0
battle_win[battle_win.index=="Shuckle"]


# Next, to complete data of 17 pokemons didn't experience a battle, untize pokemon dataframe with winning ratio. 

# In[ ]:


id_dict = dict(zip(pokemon['Name'], pokemon['#']))
battle_win["Name"] = battle_win.index
battle_win["#"] = battle_win["Name"].replace(id_dict)
ratio_dict = dict(zip(battle_win['#'], battle_win['ratio']))
pokemon["ratio"] = pokemon["#"].replace(ratio_dict)
pokemon.head()


# In[ ]:


nobattle_pokemon = pokemon[pokemon["ratio"]>1]
print ("There are {} pokemons have NaN ratio.".format(len(nobattle_pokemon.index)))
nobattle_pokemon[["#", "Name", "ratio"]]


# #63 pokemon's name is NaN. Searching by Internet, #63 pokemon is "Primeape".

# <img src="https://lh3.googleusercontent.com/YrLpyLDBZzJs7cnFdE4Yv8cIzbAKbM4v_-ZAEIbkj0faWK7yXIZ2-dJuVqTykNLRR3oDmWENIbmYO1yPak1OdLpESmucqjJiMrmSrQ7LI6yMn38--wkv2y0Q69OWFHPMBaatY6tTxoLg3ZuOq6KziO6TD3iXTWdVVuw3bnRwzwpyXmfSNKFhXGn5hQJ32FW3MvulNHCpD_b6nVCd6K2BOoSrp3e3MqEobJQJxi7sTcNgi1s19EcxZRuKY4RLOf67BPTt1iPWzXjLwA15M8jfgLNTeIm55SCg3iYWAXdCDUfjyONnB-es8-W9KviZt3nPFIbcaugeXa90YIMdfaqpdveDcAmuhuVuhxAtQuUEaCJgIGrNkWUnGdjzHEXEZq0h6BzTq6IWwl8X3FqlTXzmllWbuS_0BJY6RokZNUS4_IOI8JH84cogSmDPtIm9QERwO-cXhiFmXaeMT8tNs6aqMbbjqbzfmyZ1z7nSF8MqpFDpqp3qlIo4Mk2BObhsTwVEioRAgOa9Nx0YLR8oMcQKWmwCBpdcPaCVEOrsAZzB2PK-VhPSp_LFSau_9---cV7Mf-ov8PbdFdFZ7FNdyP17mqC6A4adiixsfFq4C1gxtQ=s431-no" width=30%>

# In[ ]:


pokemon.loc[62, "Name"] = "Primeape"
pokemon[pokemon["Name"]=="Primeape"][["#", "Name", "ratio"]]


# To complete winning ratio, look correration between base stats total and winning ratio.

# In[ ]:


battle_pokemon = pokemon[pokemon["ratio"] <= 1]
sns.lmplot(x="stats_sum", y="ratio", data=battle_pokemon)


# By using this correlation, calculate winning ratio of 17 pokemons doesn't have ratio.

# In[ ]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(battle_pokemon["stats_sum"].values.reshape(-1, 1), battle_pokemon["ratio"].values.reshape(-1, 1))
nobattle_pokemon["ratio"] = linreg.predict(nobattle_pokemon["stats_sum"].values.reshape(-1, 1))
nobattle_pokemon[["#", "Name", "ratio"]]


# ##### 2-2. Preparing DataFrame for predicting
# Add pokemon's specifications to battle combination dataframe. 

# In[ ]:


combats_add_data = combats.copy()
type_dict = dict(zip(pokemon['#'], pokemon['Type']))
hp_dict = dict(zip(pokemon['#'], pokemon['HP']))
attack_dict = dict(zip(pokemon['#'], pokemon['Attack']))
defense_dict = dict(zip(pokemon['#'], pokemon['Defense']))
spattack_dict = dict(zip(pokemon['#'], pokemon['Sp. Atk']))
spdefense_dict = dict(zip(pokemon['#'], pokemon['Sp. Def']))
speed_dict = dict(zip(pokemon['#'], pokemon['Speed']))
stats_sum_dict = dict(zip(pokemon['#'], pokemon['stats_sum']))
ratio_dict = dict(zip(pokemon['#'], pokemon['ratio']))
combats_add_data["First_pokemon_type"] = combats_add_data["First_pokemon"].replace(type_dict)
combats_add_data["First_pokemon_hp"] = combats_add_data["First_pokemon"].replace(hp_dict)
combats_add_data["First_pokemon_attack"] = combats_add_data["First_pokemon"].replace(attack_dict)
combats_add_data["First_pokemon_defense"] = combats_add_data["First_pokemon"].replace(defense_dict)
combats_add_data["First_pokemon_spattack"] = combats_add_data["First_pokemon"].replace(spattack_dict)
combats_add_data["First_pokemon_spdefense"] = combats_add_data["First_pokemon"].replace(spdefense_dict)
combats_add_data["First_pokemon_speed"] = combats_add_data["First_pokemon"].replace(speed_dict)
combats_add_data["First_pokemon_stats"] = combats_add_data["First_pokemon"].replace(stats_sum_dict)
combats_add_data["First_pokemon_ratio"] = combats_add_data["First_pokemon"].replace(ratio_dict)
combats_add_data["Second_pokemon_type"] = combats_add_data["Second_pokemon"].replace(type_dict)
combats_add_data["Second_pokemon_hp"] = combats_add_data["Second_pokemon"].replace(hp_dict)
combats_add_data["Second_pokemon_attack"] = combats_add_data["Second_pokemon"].replace(attack_dict)
combats_add_data["Second_pokemon_defense"] = combats_add_data["Second_pokemon"].replace(defense_dict)
combats_add_data["Second_pokemon_spattack"] = combats_add_data["Second_pokemon"].replace(spattack_dict)
combats_add_data["Second_pokemon_spdefense"] = combats_add_data["Second_pokemon"].replace(spdefense_dict)
combats_add_data["Second_pokemon_speed"] = combats_add_data["Second_pokemon"].replace(speed_dict)
combats_add_data["Second_pokemon_stats"] = combats_add_data["Second_pokemon"].replace(stats_sum_dict)
combats_add_data["Second_pokemon_ratio"] = combats_add_data["Second_pokemon"].replace(ratio_dict)


def calcTypeRelation(combats_add_data):
    r0 = 1
    first_type1 = combats_add_data["First_pokemon_type"].split("/")[0]
    first_type2 = combats_add_data["First_pokemon_type"].split("/")[1]
    second_type1 = combats_add_data["Second_pokemon_type"].split("/")[0]
    second_type2 = combats_add_data["Second_pokemon_type"].split("/")[1]
    if first_type2 != "None" and second_type2 != "None":
        r1 = df_type_relation[first_type1][second_type1]
        r2 = df_type_relation[first_type1][second_type2]
        r3 = df_type_relation[first_type2][second_type1]
        r4 = df_type_relation[first_type2][second_type2]
        r = r0 * r1 * r2 * r3 * r4
    elif first_type2 != "None" and second_type2 == "None":
        r1 = df_type_relation[first_type1][second_type1]
        r3 = df_type_relation[first_type2][second_type1]
        r = r0 * r1 * r3
    elif first_type2 == "None" and second_type2 != "None":
        r1 = df_type_relation[first_type1][second_type1]
        r2 = df_type_relation[first_type1][second_type2]
        r = r0 * r1 * r2
    elif first_type2 == "None" and second_type2 == "None":
        r1 = df_type_relation[first_type1][second_type1]
        r = r0 * r1
    return r

combats_add_data["Relation"] = combats_add_data.apply(lambda x: calcTypeRelation(x), axis = 1)
combats_add_data["First_win"] = combats_add_data.apply(lambda x: 1 if x["First_pokemon"]==x["Winner"] else 0, axis=1)
noneed_cols = ["First_pokemon", "Second_pokemon", "Winner", "First_pokemon_type", "Second_pokemon_type"]
combats_add_data = combats_add_data.drop(noneed_cols, axis=1)
combats_add_data.head()


# Using this dataframe, make model for predicting. Before that, split this dataframe to train data and test data.

# In[ ]:


from sklearn.model_selection import train_test_split

X = combats_add_data.drop("First_win", axis=1)
y = combats_add_data["First_win"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
print("X_train.shape = " + str(X_train.shape))
print("X_test.shape = " + str(X_test.shape))
print("y_train.shape = " + str(y_train.shape))
print("y_test.shape = " + str(y_test.shape))


# ##### 2-3. Regression, Classifier and Verification

# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log = round(logreg.score(X_test, y_test)*100, 2)
acc_log


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_knn = round(knn.score(X_test, y_test) * 100, 2)
acc_knn


# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
acc_perceptron = round(perceptron.score(X_test, y_test) * 100, 2)
acc_perceptron


# In[ ]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 
              'Naive Bayes', 'Perceptron', 'Decision Tree', 'Random Forest'],
    'Score': [acc_log, acc_knn, acc_gaussian, acc_perceptron, 
              acc_decision_tree, acc_random_forest]})
models.sort_values(by='Score', ascending=False)


# RandomForest is best argorithm, and scored 95%. 
# How much effective are factors?

# In[ ]:


effective = pd.DataFrame()
effective["feature_name"] = X.columns.tolist()
effective["feature_importance"] = random_forest.feature_importances_
effective.sort_values("feature_importance",ascending=False)


# Most effective factor is Speed. 

# In[ ]:


pokemon.sort_values("Speed",ascending=False).head(10)


# <img src="https://lh3.googleusercontent.com/SgsAM_mK8zpBbY0_vGgBVpXOUXnOowwXuBnOZa760XDXhssWirpjfucz-aaYYu_eWYoFjik65OlgBI1TqLFL1t5Eljj8LE817pLxCzDZ_kk4VQ_mDGjWCcX0O4Gw7S84a2OEBMi3fednyXyw5EW5uz0SJVa0HJo6dVR9MmlstlX8rz2DmfAddwplsTIjZqkEr56uWP4FJPzwwJX2F1KBhctS6-UJEUPuQska34YeckkAaO6CXIXCoNataRCe8Rwc6Fq_iTk-Q_99rcEldW5bafahjfpgAuM62JTZamdOJoOlMgukgLSYezYlAliYP2dQn58uU_LBZxEB9AIUpy-D9ONkQLqvqK3f1ZeODHBLuQo46tZn_kMxIaxCIZN-a0vDl6-Q40tSS6BVzo2HQLK_txRarjZzPM8tgJTyOPeIO6vUdakgU8SJ_gDOKjhXJrA5NZ3hvo3hVtPfE9B6GLRnNj1WsZgMwJS8KfLNi6b6mUqjFGOgfoZqLjz8lYz3T8L1LRgGl6d-owwhJgbpltWM6j0IHtTSpn4t7cENXBopBN70Iv4j-ENHccTJrzd01a_XtN2Kx5pqCzUrxi1CfOo7QjFso1F3uWrpG5woIHaQ4Q=w593-h281-no">

# Best 10 fastest pokemons are above.

# ##### 2-4. Predicting

# In[ ]:


tests = pd.read_csv('../input/tests.csv')
tests_add_data = tests.copy()
tests_add_data["First_pokemon_type"] = tests_add_data["First_pokemon"].replace(type_dict)
tests_add_data["First_pokemon_hp"] = tests_add_data["First_pokemon"].replace(hp_dict)
tests_add_data["First_pokemon_attack"] = tests_add_data["First_pokemon"].replace(attack_dict)
tests_add_data["First_pokemon_defense"] = tests_add_data["First_pokemon"].replace(defense_dict)
tests_add_data["First_pokemon_spattack"] = tests_add_data["First_pokemon"].replace(spattack_dict)
tests_add_data["First_pokemon_spdefense"] = tests_add_data["First_pokemon"].replace(spdefense_dict)
tests_add_data["First_pokemon_speed"] = tests_add_data["First_pokemon"].replace(speed_dict)
tests_add_data["First_pokemon_stats"] = tests_add_data["First_pokemon"].replace(stats_sum_dict)
tests_add_data["First_pokemon_ratio"] = tests_add_data["First_pokemon"].replace(ratio_dict)
tests_add_data["Second_pokemon_type"] = tests_add_data["Second_pokemon"].replace(type_dict)
tests_add_data["Second_pokemon_hp"] = tests_add_data["Second_pokemon"].replace(hp_dict)
tests_add_data["Second_pokemon_attack"] = tests_add_data["Second_pokemon"].replace(attack_dict)
tests_add_data["Second_pokemon_defense"] = tests_add_data["Second_pokemon"].replace(defense_dict)
tests_add_data["Second_pokemon_spattack"] = tests_add_data["Second_pokemon"].replace(spattack_dict)
tests_add_data["Second_pokemon_spdefense"] = tests_add_data["Second_pokemon"].replace(spdefense_dict)
tests_add_data["Second_pokemon_speed"] = tests_add_data["Second_pokemon"].replace(speed_dict)
tests_add_data["Second_pokemon_stats"] = tests_add_data["Second_pokemon"].replace(stats_sum_dict)
tests_add_data["Second_pokemon_ratio"] = tests_add_data["Second_pokemon"].replace(ratio_dict)
tests_add_data["Relation"] = tests_add_data.apply(lambda x: calcTypeRelation(x), axis = 1)
noneed_cols = ["First_pokemon", "Second_pokemon", "First_pokemon_type", "Second_pokemon_type"]
tests_add_data = tests_add_data.drop(noneed_cols, axis=1)
y_predict = random_forest.predict(tests_add_data)
data = {"First_pokemon": tests["First_pokemon"], "Second_pokemon": tests["Second_pokemon"], "First_win": y_predict}
submission = pd.DataFrame(data=data, columns=["First_pokemon", "Second_pokemon", "First_win"])
winner = pd.DataFrame(submission.apply(lambda x: x["First_pokemon"] if x["First_win"]==1 else x["Second_pokemon"], axis=1), columns=["Winner"])
submission = pd.concat([submission, winner], axis=1)
submission = submission.drop(["First_win"], axis=1)
submission


# In[ ]:




