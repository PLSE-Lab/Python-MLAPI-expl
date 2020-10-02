#!/usr/bin/env python
# coding: utf-8

# **NFL Draft Analysys**
# 
# What goes into a successful draft? Are there patterns of success which GMs are regrettably unable to identify? I'll go through the draft data of the last 30 years and try to uncover patterns of success and general behavior in an attempt to gain insight on one of the biggest gambles there is: the NFL Draft. 

# In[ ]:


#Import libraries

import numpy as np 
import pandas as pd 

#Read data, count rows
data = pd.read_csv('../input/nfl_draft.csv')
rows = len(data.index)


# In[ ]:


#Lets view the top of the data to see the format

data.head(1)


# We see there's a bunch of different categories. Many of these we will only know at the end of one's career, so when we attempt to predict a draft picks success, we'll have to chose metrics which we know on draft day. 

# In[ ]:


#Now lets view the bottom to ensure nothing bizarre happens in the data

data.tail(1)


# In[ ]:


#We see sacks are turned into Colleges at some point

#let's move sacks over to colleges where applicable with a function


#Our function will switch sacks with University, where applicable
def switch_sacks(df): 
    #Copy the dataframe
    df = df.copy(deep=True)
    #Get the number of rows
    nrows = len(df.index) 
    
    #Create a test of if a string is a number
    #We'll use this to make sure that the value for sacks is not a number
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    # Now we make sure that both the College isn't labeled and he has no sacks
    # A couple players didnt go to college, so its important to have both
    for i in range(nrows):
        if isinstance(df.loc[i]["College/Univ"], str) == False and is_number(df.loc[i]["Sk"]) == False: 
            df.set_value(i, "College/Univ", df.loc[i]["Sk"])
            df.set_value(i, "Sk", np.nan)
            
    return df


# In[ ]:


#Lets make this data clean and check all of the colleges

newdata = switch_sacks(data)


# **Adding Additional Information**
# 
# Currently, we have information on the school, but not the conference the school is from. Similarly, we have information on the team drafted, but not their conference or division. Seeing as some divisions may be stronger than others, perhaps there are certain fits for certain players. Similarly, it may make sense that a Pac 12 player would be drafted by an NFC West team. They will not only play in a familiar environment, but the coaches will be able to get more looks at them. As a result, they'll be able to better diagnose their ability and fit within the team/system.

# In[ ]:


#Lets add a column for conference
#We'll only use the Power 5

def add_cfb_conf(df): 
    conf =[]
    nrows = len(df.index) 
    
    pac12 = ("Stanford", "California", "Arizona St.", "Arizona", "Washington", 
          "Washington St.", "Oregon", "Oregon St.", "USC", "UCLA", "Utah", "Colorado")
    
    big12 = ("Oklahoma", "Oklahoma St.", "TCU", "Baylor", "Iowa St.", "Texas", "Kansas", 
            "Kansas St.", "West Virginia", "Texas Tech")
    
    b1g = ("Northwestern", "Michigan", "Michigan St.", "Iowa", "Ohio St.", "Purdue", 
          "Indiana", "Rutgers", "Illinois", "Minnesota", "Penn St.", "Nebraska", "Maryland", 
          "Wisconsin")
    
    acc = ("Florida St.", "Syracuse", "Miami", "North Carolina", "North Carolina St.", 
          "Duke", "Virginia", "Virginia Tech", "Boston College", "Clemson", "Wake Forest",
          "Pittsburgh", "Louisville", "Louisville", "Georgia Tech")
    
    sec = ("Alabama", "Georgia", "Vanderbilt", "Kentucky", "Florida", "Missouri", 
          "Mississippi", "Mississippi St.", "Texas A&M", "Louisiana St.", "Arkansas", 
          "Auburn", "South Carolina", "Tennessee")
    
    for i in range(nrows): 
        if df.loc[i]["College/Univ"] in pac12: 
            conf.append("Pac 12")
            
        elif df.loc[i]["College/Univ"] in big12:
            conf.append("Big 12")
            
        elif df.loc[i]["College/Univ"] in b1g:
            conf.append("Big 10")
            
        elif df.loc[i]["College/Univ"] in acc:
            conf.append("ACC")
            
        elif df.loc[i]["College/Univ"] in sec: 
            conf.append("SEC")
        
        else: 
            conf.append("Not Power 5")
    return conf


# In[ ]:


conf = add_cfb_conf(newdata)


# In[ ]:


newdata['CFB_Conference'] = conf


# In[ ]:


newdata


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


# Let's now take a look and see if any conferences in particular are favored. We see that non-power 5 conferences have the most drafted players by a large margin. That being said, Power 5 picks do outnumber non-power 5 picks. We should also know that missing data points and perhaps misclassified data points will fall under the non-power 5 label. This may make it artificially inflated. 
# 
# What GM's should take note of from this however is that you can most certainly find loads of talent from lesser known schools. Take a look at Kareem Hunt, he currently leads the league in rushing and was from Toledo. He was a late round pick, and while I don't mean to say that he was a better selection than say Christian McCaffrey or Leonard Fournette, it is now clear that he is going to have a far better value. 

# In[ ]:


sns.countplot(x = "CFB_Conference", data = newdata, palette = "Greens_d")


# **Round Distribution**
# 
# We see that these non power 5 picks happen mainly in the later rounds, with a large emphasis on the 7th round. We can also see that SEC players are picked most often in the 1st round. These two notions are generally understood. The SEC is perenially the best conference and non-power 5 teams are just simply not up to par. These players are elite draft talents and, as we can see, are held in high regard by GMs

# In[ ]:


sns.violinplot(x = "CFB_Conference", y="Rnd", data = newdata, palette = "husl")


# In[ ]:


newdata.loc[newdata['Rnd'] == 8].head(1)


# **Rounds 1 -7**
# 
# Fun Fact: Up until 1993, there were more than 7 rounds in the NFL Draft. This may come as a surprise to us millenials, many of which were born after this data. To get a better look at the rounds that now matter, I'll cut out 8-12. 
# 
# We see there are slightly different distributions for the Power 5 conferences, but there's definitely still the glaring difference of the non-power 5 teams. 

# In[ ]:


sns.violinplot(x = "CFB_Conference", y="Rnd", data = newdata.ix[:5957][:], palette = "husl")


# In[ ]:


Rd1 = newdata.loc[newdata['Rnd'] == 1]


# **The Money Round**
# 
# First round picks are incredibly important. It's where you build your team and if you have the fortune of multiple first round picks, you are usually putting yourself in a position to perform well in the future. I'll do some analysis of how teams address their most glaring needs in the first rounds. 
# 
# Often times, teams will reach for QBs in the first round, in an attempt to solidify their franchise for years to come. This can lead to some interesting results. 

# In[ ]:


Rd1


# In[ ]:


import matplotlib.pyplot as plt

Positions = Rd1.groupby(['Pos']).size()

Positions.index


# In[ ]:


labels = Positions.index
sizes = Positions
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0)  # only "explode" the QBs

fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Wait.... I thought QBs were the belle of the ball? As one can see, QBs make up an incredibly small portion of overall first round picks. Many of these picks are spent on the other glamorous positions in the sport. Those being DE, DB, WR, and RB. You also see DTs and LBs both having more first round selections than QBs, which is also quite interesting.
# 
# Sure, you may say that there are many of all of these positions on the field at one time and only one QB, but it goes to show that realistically, teams can address other needs in the first round. They consistently do so by improving skill positions and defense over QBs. This pattern usually should result in success if the old adage of "defense wins championships" holds true. 

# In[ ]:


Top10 = Rd1.loc[Rd1['Pick'] < 11]

Positions = Top10.groupby(['Pos']).size()

labels = Positions.index
sizes = Positions
labels

explode = (0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0)  # only "explode" the QBs

fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# *Not so fast my friend*
# 
# When you examine top 10 picks, you'll see that QBs are one of the most significant positions. This would make sense as well. Teams that perform poorly will own these top 10 picks. If you're a bad team, you likely have a bad quarterback. These types of teams love to invest in the future with a 'franchise quarterback.'  It would be interesting to analyze where these picks land one in the draft order in the following year. This would be incredibly effective in gaugin the quality of picks for a team, but for now, we'll focus on other metrics. 

# In[ ]:


newdata.groupby(['Tm']).size()


# In[ ]:


def add_div_conf(df):
    div = []
    conf = []
    nrows = len(df.index)
    
    afcsouth = ['IND', 'JAX', 'TEN', 'HOU']
    afcnorth = ['PIT', 'BAL', 'CIN', 'CLE']
    # Must account for the moves of the raiders
    afcwest = ['RAI', 'OAK', 'SDG', 'DEN', 'KAN']
    afceast = ['BUF', 'MIA', 'NWE', 'NYJ']
    
    nfcsouth = ['CAR', 'NOR', 'TAM', 'ATL']
    nfceast = ['WAS', 'PHI', 'NYG', 'DAL']
    # Must account for moves of cardinals and rams
    nfcwest = ['SFO', 'STL', 'RAM', 'PHO', 'ARI', 'SEA']
    nfcnorth = ['CHI', 'GNB', 'MIN', 'DET']
    
    
    for i in range(nrows):
        
        
        if df.loc[i]['Tm'] in afcsouth: 
            div.append('South')
            conf.append('AFC')
        
        elif df.loc[i]['Tm'] in nfcsouth: 
            div.append('South')
            conf.append('NFC')
            
        elif df.loc[i]['Tm'] in afcwest: 
            div.append('West')
            conf.append('AFC')
        
        elif df.loc[i]['Tm'] in nfcwest: 
            div.append('West')
            conf.append('NFC')
            
        elif df.loc[i]['Tm'] in afceast: 
            div.append('East')
            conf.append('AFC')
            
        elif df.loc[i]['Tm'] in nfceast: 
            div.append('East')
            conf.append('NFC')
        
        elif df.loc[i]['Tm'] in afcnorth: 
            div.append('North')
            conf.append('AFC')
        
        elif df.loc[i]['Tm'] in nfcnorth: 
            div.append('North')
            conf.append('NFC')
            
        else: 
            div.append(float('nan'))
            conf.append(float('nan'))
        
    return div, conf


# In[ ]:


div, conf = add_div_conf(newdata)


# In[ ]:


newdata['Conf'] = conf
newdata['Div'] = div


# In[ ]:


newdata


# **Hometown Heroes**
# 
# A draft day story everyone loves to see is the local legend getting picked up by the local NFL team. He's always been the darling of the city and now can be there for his whole football career. It's a storybook draft. 
# 
# We can see below that often times the favorite conference of an NFL is the most local one. You see the West really likes the Pac 12, whereas the North and East have a particular love for the Big 10. This makes perfect sense. As I've previously stated, you know these local players better and are making a more educated guess on how well they'll perform. 

# In[ ]:


from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(20,10))
sns.countplot(y="Div", hue="CFB_Conference", data=newdata.loc[newdata['CFB_Conference']!= "Not Power 5"], palette="hls");


# In[ ]:


#Let's fix up a missing year of rounds

newdata['Rnd'].isnull().sum().sum()


# In[ ]:


def fix_rds(df): 
    nrow = len(df.index)
    
    for i in range(nrow): 
        if df.loc[i]['Year'] == 1993: 
            if df.loc[i]['Pick'] >= 1 and df.loc[i]['Pick'] <= 28: 
                df.set_value(i, 'Rnd', 1)
            elif df.loc[i]['Pick'] >= 29 and df.loc[i]['Pick'] <= 56: 
                df.set_value(i, 'Rnd', 2)
            elif df.loc[i]['Pick'] >= 57 and df.loc[i]['Pick'] <= 84: 
                df.set_value(i, 'Rnd', 3)
                
            elif df.loc[i]['Pick'] >= 85 and df.loc[i]['Pick'] <= 112: 
                df.set_value(i, 'Rnd', 4)
                
            elif df.loc[i]['Pick'] >= 113 and df.loc[i]['Pick'] <= 140: 
                df.set_value(i, 'Rnd', 5)
            
            elif df.loc[i]['Pick'] >= 141 and df.loc[i]['Pick'] <= 168: 
                df.set_value(i, 'Rnd', 6)
            
            elif df.loc[i]['Pick'] >= 169 and df.loc[i]['Pick'] <= 196: 
                df.set_value(i, 'Rnd', 7)
            
            elif df.loc[i]['Pick'] >= 197 and df.loc[i]['Pick'] <= 224: 
                df.set_value(i, 'Rnd', 8)
    return df

newdata = fix_rds(newdata)


# In[ ]:


# Lets check in all of our pertinent categories for missing values and replace them

newdata['Rnd'].isnull().sum().sum()


# In[ ]:


newdata['Pick'].isnull().sum().sum()


# In[ ]:


newdata['Tm'].isnull().sum().sum()


# In[ ]:


newdata['Pos'].isnull().sum().sum()


# In[ ]:


newdata['Position Standard'].isnull().sum().sum()


# In[ ]:


newdata['Age'].isnull().sum().sum()

#Way too many missing values, let's just avoid this section entirely. Realistically, the age 
#distribution won't be of utmost concern anyways. 


# In[ ]:


newdata['College/Univ'].isnull().sum().sum()

#Lets go ahead and replace all of these values with None. Perhaps these students didn't attend 
# college, perhaps it was overseas. 


# In[ ]:


newdata['College/Univ'].fillna("Other", inplace=True)

newdata['College/Univ'].isnull().sum().sum()


# In[ ]:


newdata['CFB_Conference'].isnull().sum().sum()


# In[ ]:


newdata['Conf'].isnull().sum().sum()


# In[ ]:


newdata['Div'].isnull().sum().sum()


# **Fitting a Model**
# 
# An important part of this model will be choosing the parameters which we know on draft day. Ideally, I'd like to have all of their combine statistics, college stats, and record from college. Life isn't perfect, so I don't. But we can know which conference they played in, which position they are, what pick they would be, what team they would play for, etc. From this, hopefully we'll be able to see what is most important in this process. 

# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

train, test = train_test_split(newdata, test_size=0.2)


def encode_features(df_train, df_test):
    features = ['Year', 'Pick', 'Rnd', 'College/Univ', 'Tm', 'Div', 'Conf', 'Pos', 
                'Position Standard', 'CFB_Conference']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(train, test)
data_train.head()


# **Adding Zeros for DNP**
# 
# If a player never played, their G and CarAV will be zero. In this data, it is omitted so we'll have to replace them with zeros. 
# 

# In[ ]:


data_train['G'].isnull().sum()


# In[ ]:


data_train['G'].fillna(0, inplace=True)
data_train['CarAV'].fillna(0, inplace = True)

data_test['G'].fillna(0, inplace=True)
data_test['CarAV'].fillna(0, inplace = True)


# In[ ]:


from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Player_Id', 'Player', 'First4AV', 'Age', 'To', 'AP1',
                        'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Pass_Att', 
                        'Pass_TD', 'Pass_Int', 'Rush_Att', 'Rush_Yds', 'Rush_TDs',
                        'Rec', 'Rec_Yds', 'Rec_Tds', 'Tkl', 'Def_Int', 'Sk',
                        'Pass_Yds', 'Unnamed: 32'], axis=1)
y_all = data_train['CarAV']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# In[ ]:


y_train.isnull().sum()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
mdl = RandomForestRegressor(n_estimators = 50, criterion = 'mae', max_features = "log2")

# Fit the best algorithm to the data. 
mdl.fit(X_train, y_train)


# In[ ]:


predictions = mdl.predict(X_test)


# **Model Evaluation**
# 
# We see the rsq is 30, which isn't great, but considering this stuff is nearly impossible to predict, this is a great result. There is so much that goes into these players' success and it is tough to be categorized. Teams change from year to year, injuries come out of no where, players may have 'intagibles' issues, etc. 
# 
# Let's cross validate, then examine what is important in these models.

# In[ ]:


from sklearn.metrics import r2_score

r2_score(y_test, predictions)


# In[ ]:


residuals = predictions - y_test
plt.hist(residuals, bins = 30, range = (-50, 50))

# The distribution is left-skewed, as we'll have a handful of players that perform very well
# This would follow the idea of power law distributions, where many have little success
# And few have much success


# In[ ]:


from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score

def run_kfold(mdl):
    kf = KFold(6748, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        mdl.fit(X_train, y_train)
        predictions = mdl.predict(X_test)
        accuracy = r2_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(mdl)


# In[ ]:


importance = mdl.feature_importances_


# In[ ]:


importance


# In[ ]:


X_train.columns


# In[ ]:



objects = X_train.columns
y_pos = np.arange(len(objects))


# **What's the Takeaway?**
# 
# We see that most importantly, you have your pick order. Chances are, your number one pick should succeed. But that being said, that only accounts for .27 of the importance. Incredibly important factors are college, conference, division, team, draft year and more. 
# 
# What's this mean? It means that you can find talent throughout the draft. Very often, GMs may cover their own behinds by choosing highly touted prospects or even worse, trading picks to move up in the draft. The Dallas Cowboys built their dynasty in the 80s & 90s by trading for picks and the Washington Redskins made one of the worst deals of all time to move up for RG III. What GMs need to understand is that the draft is filled with talent, top to bottom. The way to build an effective team is through the draft and having a bevy of picks. 
# 
# Once you have these picks, be judicious. Talent has a part to do with it, but culture fit is a much bigger factor. It's the same reason that Stanford, over the past 10 years, can turn two and three star recruits into Rose Bowl Champions. Stanford's record over the last 10 years trails only Alabama. This is because they're able to find the right players, not the most highly rated ones. Five-star recruits, such as Barry Sanders Jr., have fizzled there. 
# 
# 

# In[ ]:


plt.barh(y_pos, importance, align='center', alpha=0.5)

plt.yticks(y_pos, objects)

plt.xlabel('Importance')
plt.title('Feature Importance')
 
plt.show()


# **Let's try it for games played**
# 
# We see that games played is a bit harder to predict. This is for a variety of reasons. One, the scale of games played is far higher. Another is that it, in general, is harder to predict. Injuries happen and of late, players retire early due to health concerns. This is only going ot happen more and more in the future, cutting careers short and making this metric even tougher to predict. 
# 
# That being said, you see that the same predictors are of almost the exact same importance. It goes to show that if you have a long career, you are most certainly driving value to a franchise. There's a reason they're hiring you after all. 

# In[ ]:


X_all = data_train.drop(['Player_Id', 'Player', 'First4AV', 'Age', 'To', 'AP1',
                        'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Pass_Att', 
                        'Pass_TD', 'Pass_Int', 'Rush_Att', 'Rush_Yds', 'Rush_TDs',
                        'Rec', 'Rec_Yds', 'Rec_Tds', 'Tkl', 'Def_Int', 'Sk',
                        'Pass_Yds', 'Unnamed: 32'], axis=1)
y_all = data_train['G']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=22)


# In[ ]:


mdl.fit(X_train, y_train)


# In[ ]:


predictions = mdl.predict(X_test)
r2_score(y_test, predictions)


# In[ ]:


#You'll see the same left-skew, which follows the power law idea

residuals = predictions - y_test
plt.hist(residuals, bins = 30, range = (-150, 150))
plt.title('Residuals: Games Played')


# In[ ]:


run_kfold(mdl)


# In[ ]:


importance1 = mdl.feature_importances_
importance1 - importance


# In[ ]:


plt.barh(y_pos, importance1, align='center', alpha=0.5)

plt.yticks(y_pos, objects)

plt.xlabel('Importance')
plt.title('Feature Importance: Games Played')
 
plt.show()


# In[ ]:


newdata.sort_values(['G'], ascending=[False]).head(10)


# **Gold Jackets**
# 
# Let's add a HOF section and do a quick analysis. We see that basically all of these players are picked in the first round and are picked from non-power 5 schools. 
# 
# Again, GMS: look for talent in non-traditional places and stockpile those picks. Particularly first round picks. It's tough to pick winners, but if you have many first rounders, you are going to have a great shot at getting HOF caliber players. Similarly, look for synergies between the player and your team/culture. These fits are incredibly tough to characterize, but an effective GM will be able to identify and capitalize on these players. 

# In[ ]:


newdata.sort_values(['CarAV'], ascending=[False]).head(10)


# In[ ]:


def hof(df): 
    nrow = len(df.index)
    hall = []
    for i in range(nrow):
        if (df['Player'].str[-3:][i] == "HOF"): 
            hall.append(1)
        else: 
            hall.append(0)
    
    return hall
newdata['HOF'] = hof(newdata)


# In[ ]:


thehall = newdata.loc[newdata['HOF'] == 1]
thehall.head(3)


# In[ ]:


Rounds = thehall.groupby(['Rnd']).size()

labels = Rounds.index
sizes = Rounds
explode = (0.3, 0, 0, 0, 0, 0)  # only "explode" the QBs

fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Hall of Famers by round drafted')

plt.show()


# In[ ]:


Con = thehall.groupby(['CFB_Conference']).size()
labels = Con.index
sizes = Con
explode = (0, 0, 0, 0, 0, 0.3)  # only "explode" the QBs

fig1, ax1 = plt.subplots(figsize=(20,10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Hall of Famers by Conference')

plt.show()


# In[ ]:




