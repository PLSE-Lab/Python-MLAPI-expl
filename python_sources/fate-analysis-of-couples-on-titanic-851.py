#!/usr/bin/env python
# coding: utf-8

# This notebook goes back in time to develop a picture of what event were actually taking place on the Titanic at the time of the sinking. What strategies were available for surivival? How were they implemented? Let's take a look.

# In[ ]:


import numpy as np
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore")

def findTitle(row):
    val = row["Name"].split(",")[1].split(".")[0].strip() 
    if ((val in ["Dr"]) & (row["Sex"] == "female")):
        val = 'woman'
    if ((val in ["Mr"]) & (row["Age"] <= 13)):
        val = 'boy'
    if val in ["Capt","Don","Major","Col","Rev","Dr","Sir","Mr","Jonkheer"]:
        val = 'man'
    if val in ["Dona","the Countess","Mme","Mlle","Ms","Miss","Lady","Mrs"]:
        val = 'woman'
    if val in ["Master"]:
        val = 'boy'
    return val

def addAverage(row):
    try:
        if (row.Family_Id in gs['Family_Id'].values):
            val = gs[gs.Family_Id==row.Family_Id]['Avg_SurvivalRate']
        else:
            val=0
    except:
        val=0
    return float(val)

def printAccuracy():
    print (fd.loc[(fd.Predict == fd.Survived_Actual)].count()['PassengerId']/1309)

survival_data = pd.read_csv('../input/titanic-groups/titanic.csv')
fd = pd.read_csv('../input/real-groups/real_groups.csv')

for i, name in enumerate(survival_data['name']):
    if '"' in name:
        survival_data['name'][i] = re.sub('"', '', name)
        
for i, name in enumerate(fd['Name']):
    if '"' in name:
        fd['Name'][i] = re.sub('"', '', name)

survived = []

for name in fd['Name']:
    survived.append(int(survival_data.loc[survival_data['name'] == name]['survived'].values[-1]))

fd['Survived_Actual'] = survived
fd['Predict'] = -9
fd.rename(columns={'Couple_ID':'Family_Id'},inplace = True)


# We are going to start our journey before embarkation: what do we know about people who travelled on the Titanic? A great many things.  We know that travel is often with family members, or with friends, and that the group you start your travels with is likely the group you end your travels with. Importantly, we suspect this may be true even if the 'end of your travels' results in you and your group's demise. We know that some family groups had maids, servants, and nannies to accompany them on their voyage, although we aren't fully aware of the cohesiveness of these non-family members to the group.  We also know that sometimes families travel with other families, or meet other famliies on board, and spend a large amount of their time together on the trip as a group.  Again, it is unclear how cohesive these ties are.
# 
# We also know quite bit about people in general.  We know that survival is a strong motivating factor, but this is not the greatest.  The strongest motivator is  love, particularly love of one's family.   We know that people make rules and if there is a sufficent power dynamic then these rules are followed by others.  So for example if a parent tells a child to get into a lifeboat, the child will do it. Simiarly, if an armed member of the White Star crew limits access to the lifeboats to women and children, then probably only women and children will be aboard that lifeboat. We also know that, in the absence of authority, rules can be selectively followed and that those who make the rules can change them at a moment's notice and at whim. Finally, we know that people have a tendency to follow patterns: we can be classist, sexist, and xenophobic.
# 
# As such, the basis for our model is one of groups, specifically the groups that were formed immediately before attempting to find a lifeboat.  Many notebooks use this principle and apply such items as Surname, Cabin number, and Ticket number as delineating factors in group formation. This book supplements these groups with eyewitness accounts, taken from the Encyclopedia Titanica 
# 
# https://www.encyclopedia-titanica.org/titanic-survivors/ 
# 
# Since this is a time consuming procedure only groups of two people are explored.  The starting point was the work of Reinhard Sellmair in the Save the Families! notebook
# 
# https://www.kaggle.com/reisel/save-the-families 
# 
# Reinhard plotted out most of the relationships between families, and we are extend this information to generate 'real' family groups. To begin we will assume that all blood relatives stayed together, assuming that if the families had multiple Cabins, etc. that they all found each other before searching for a lifeboat. This seems completely natural: in a crisis sort of situation, you find the people you can trust and work together as a team to solve the problem.
# 
# An interesting facet of this problem is that there is ultimately only one solution.  Survival depends on finding a seat in a lifeboat - there is no other means to accomplish this objective. As such, only the method of acquiring the seat needs to be analyzed which greatly simplifies the problem.
# 
# Many of the groups required revision based on the eyewitness accounts and these are summarized below (two person groups only):

# In[ ]:



# Vanderplanks party all stayed together
fd.loc[(fd.PassengerId==39), 'Family_Id'] = 11
fd.loc[(fd.PassengerId==334), 'Family_Id'] = 11

# Hays Olive party stayed together
fd.loc[(fd.PassengerId==311), 'Family_Id'] = 13

# Franchi Haas Kink women stayed together, and died. This is an interesting party in that the women were staying in a 
# different cabin than their husbands and were unable to reconnect before the lifeboat search.
fd.loc[(fd.PassengerId==294), 'Family_Id'] = 131
fd.loc[(fd.PassengerId==1268), 'Family_Id'] = 131

# Hansens stayed together 
fd.loc[(fd.PassengerId==705), 'Family_Id'] = 64

# Frolichers stayed together
fd.loc[(fd.PassengerId==540), 'Family_Id'] = 85

# Skoog family was joined by two individuals
fd.loc[(fd.PassengerId==1304), 'Family_Id'] = 99
fd.loc[(fd.PassengerId==808), 'Family_Id'] = 99

# Harper family was joined by Jessie Leitch
fd.loc[(fd.PassengerId==597), 'Family_Id'] = 160

# Cardeza party included theier two servants
fd.loc[(fd.PassengerId==738), 'Family_Id'] = 70
fd.loc[(fd.PassengerId==259), 'Family_Id'] = 70

# Frauenthal party stayed together
fd.loc[(fd.PassengerId==991), 'Family_Id'] = 127

# Robins party stayed together
fd.loc[(fd.PassengerId==1296), 'Family_Id'] = 88

# Lahtinen party stayed together
fd.loc[(fd.PassengerId==418), 'Family_Id'] = 12

# No group for these
fd.loc[(fd.PassengerId==218), 'Family_Id'] = 9999
fd.loc[(fd.PassengerId==672), 'Family_Id'] = 9999

# The Kafr Mishki party.  While there are mixed survivals in this group it appears from the eyewitness accounts that 
# they at least started as a group, in whole or in part.  
fd.loc[(fd.PassengerId==1137), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==1136), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==1216), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==1216), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==1131), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==1132), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==1133), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==297), 'Family_Id'] = 200
fd.loc[(fd.PassengerId==579), 'Family_Id'] = 200


# After the 'family' groups are correctly defined, family counts and survival rates are calculated.  These will be used to pull out the two person groups and help predict survival:

# In[ ]:


groupFreq= fd.groupby('Family_Id', as_index=False).count()[['Family_Id','PassengerId']]
groupFreq.rename(columns={'PassengerId':'Family_Count'},inplace = True)

fd = pd.merge(fd, groupFreq, on = 'Family_Id', how='inner')

# Remove groups of two
groupFreq.drop(groupFreq[groupFreq.Family_Count==2].index, inplace=True)

# Now figure out what percentage of the group survived
gs = fd.groupby('Family_Id', as_index=False)['Survived'].mean()[['Family_Id','Survived']]
gs.rename(columns={'Survived':'Avg_SurvivalRate'},inplace = True)

# need to get a better classification than in the sheet, this one uses 'man, 'woman', and 'boy'
fd['Title_c'] = fd.apply(findTitle, axis=1)
fd['Family_Pct'] = fd.apply(addAverage, axis=1)


# Ok.  Now let's create some subclasses to identify what two person groups were present, these will also help us apply the social and power dynamics to the groups.   Interestingly, some groups are conspicously absent:
# 
# * man with sibling
# * woman with sibling
# * two children together
# * two strangers
# 
# Also some groups are missing or consolidated into 'child' groups:
# 
# * man with boy
# * man with girl
# * woman with boy
# * woman with girl
# * dad with son
# * dad with daughter
# * mom with son
# * mom with daughter
# 
# As it turns out, this level of detail is not required as 'with child' handles all the cases. 
# 
# What can be said about these groups?  First, there could be additional groups of two on the Titanic other than these.  However, if this is the case then all of these groups all followed the  'rules of survival' outlined below, and as such were not detected.  In spite of this the complete lack of sibling only groups of two seems remarkable; it is unclear what sociological driver could exist to create this situation. Less surprising is that there were no groups of two children or strangers.

# In[ ]:


fd.loc[(fd.Title_c == 'man') & (fd.Family_Count==2) & (fd.SpouseNumber==1),'Family_subclass'] = 'man with wife'      
fd.loc[(fd.Title_c == 'man') & (fd.Family_Count==2) & (fd.FatherNumber==1),'Family_subclass'] = 'man with dad'
fd.loc[(fd.Title_c == 'man') & (fd.Family_Count==2) & (fd.MotherNumber==1),'Family_subclass'] = 'man with mom'
fd.loc[(fd.Title_c == 'man') & (fd.Family_Count==2) & (fd.ChildrenNumber==1),'Family_subclass'] = 'man with child'

fd.loc[(fd.Title_c == 'woman') & (fd.Family_Count==2) & (fd.SpouseNumber==1),'Family_subclass'] = 'wife with husband'      
fd.loc[(fd.Title_c == 'woman') & (fd.Family_Count==2) & (fd.FatherNumber==1),'Family_subclass'] = 'woman with dad'
fd.loc[(fd.Title_c == 'woman') & (fd.Family_Count==2) & (fd.MotherNumber==1),'Family_subclass'] = 'woman with mom'
fd.loc[(fd.Title_c == 'woman') & (fd.Family_Count==2) & (fd.ChildrenNumber==1),'Family_subclass'] = 'woman with child'

fd.loc[(fd.Title_c == 'boy') & (fd.Family_Count==2) & (fd.FatherNumber==1),'Family_subclass'] = 'boy with dad'
fd.loc[(fd.Title_c == 'boy') & (fd.Family_Count==2) & (fd.MotherNumber==1),'Family_subclass'] = 'boy with mom'


# Next we identify cases that did not fit the model and, using the eyewitness acounts, try and determine the cause of the discrepancy. The model has only two basic rules.
# 
# To start, we assume all men die in the twp person groups.  It is easy to arrive at this conclusion by acknowledging the supposed rules in place ('women and children first') and the social and power dyanmics in place at the time.  Consider what was happening with these men: they were on Titanic with their own child, or their wife (often a newlywed), their mother or their dad.  They were not in a group with a stranger, a sister or a brother.  Attaining the objective is a very clear goal - get your partner to a life boat, then try and save yourself. Importantly, saving yourself will only be achieved by then breaking one of the rules that you just followed, be it social, power, or authoritarian: if you go and deliver you partner but then continue to follow the rules you will die. This is probably the fundamental factor that makes is so difficult to predict what men survived on the Titanic.
# 
# Next, we assume that all women live in the two person groups.  This rules neatly encapulates the social, power, and authoritarien dynamics.  Groups with two women have absolutely no probem getting into the boats, if they get to them. This is true for the entire time that the lifeboat loading procedure is active. The only potential problem cases are being with a man (husband, dad, or son).  However, 'son' is not an issue and in the case of 'husband' and 'dad' there are supremely powerful social factors working on their side to get you to a boat. The challenge here is the opposite of the men, in that we must determine why women die in these cases.  Interestingly, there are two possible cases: like the men, the women could break a rule to cause their own demise (and as we will see, many did) or they could follow all the rules and still die.  Again, the rule-breaking women are the challenge to applying maching learning to this group, with the slight difference that women had to choose to break the rule before getting into the lifeboat (else they would live) while men had to break the rule at the time or after their party arrived at the boat.
# 
# Amazingly the discrepancies in living/dying can be accounted using just a few categories:
# 
# Female Cases
# * Was_Stubborn - Individual was documented by eyewitnes accounts as refusing to leave husband.
# * Unknown_Cause - Cause of death unknown, not enough information.
# * Exposure - Individual reached a boat but ended up in the water
# 
# Male Cases
# * Lax_Lifeboat_Rules - Individual was able to get a seat on a lifeboat where the 'women and children first' rule was not being applied.
# * Jumped_into_Boat - Individual jumped into boat, bypassing the crew members and rules on loading procedure
# * Is_Swimmer - Individual ended up in the water but was able to swim to a boat
# 

# In[ ]:



fd.loc[(fd.PassengerId==1105),'Was_Stubborn'] = 1 # Howard, wouldnt leave her ailing husband 
fd.loc[(fd.PassengerId==1006),'Was_Stubborn'] = 1 # Straus
fd.loc[(fd.PassengerId==42),  'Was_Stubborn'] = 1 # Turpin 'not on your life'
fd.loc[(fd.PassengerId==855), 'Was_Stubborn'] = 1 # Carter wouldnt leave husband
fd.loc[(fd.PassengerId==618), 'Was_Stubborn'] = 1 # Lobb wouldnt leave husband
fd.loc[(fd.PassengerId==1011),'Was_Stubborn'] = 1 # Chapman wouldnt leave husband
fd.loc[(fd.PassengerId==1275),'Was_Stubborn'] = 1 # McNamee young and wouldnt leave husband

fd.loc[(fd.PassengerId==1274), 'UnKnown_Cause'] = 1 # Risien 
fd.loc[(fd.PassengerId==1141), 'UnKnown_Cause'] = 1 # Khalil

fd.loc[(fd.PassengerId==1251), 'Exposure'] = 1 # Lindell, actually got to a boat but died in the water

# Boat 7, first boat lowered, Pittman, possibly Ismay, and Murdoch in charge. Passengers relucant to enter the boats, strict 'women and children' rules not in force.
fd.loc[(fd.PassengerId==1179), 'Lax_Lifeboat_Rules'] = 1 # Snyder boat #7 
fd.loc[(fd.PassengerId==485),  'Lax_Lifeboat_Rules'] = 1 # Bishop boat #7 'brides and grooms may board'
fd.loc[(fd.PassengerId==98),   'Lax_Lifeboat_Rules'] = 1 # Greenfield boat #7 

# Boat 5, second boat lowered, Pittman and Murdoch in charge.  Passengers still relucant to enter boats, strict 'women and children' rules not in force.
fd.loc[(fd.PassengerId==1069), 'Lax_Lifeboat_Rules'] = 1 # Stengel boat #5 
fd.loc[(fd.PassengerId==371),  'Lax_Lifeboat_Rules'] = 1 # Harder boat #5 Ismay said ok
fd.loc[(fd.PassengerId==454),  'Lax_Lifeboat_Rules'] = 1 # Goldenberg boat #5 Ismay threw him in
fd.loc[(fd.PassengerId==622),  'Lax_Lifeboat_Rules'] = 1 # Kimball boat #5 
fd.loc[(fd.PassengerId==713),  'Lax_Lifeboat_Rules'] = 1 # Taylor boat #5 
fd.loc[(fd.PassengerId==725),  'Lax_Lifeboat_Rules'] = 1 # Chambers boat #5 

# Boat 3, third boat lowered, Murdoch in charge, strict 'women and children' rules not in force.
fd.loc[(fd.PassengerId==646),  'Lax_Lifeboat_Rules'] = 1 # Harper boat #3
fd.loc[(fd.PassengerId==249),  'Lax_Lifeboat_Rules'] = 1 # Beckwith boat #3
fd.loc[(fd.PassengerId==691),  'Lax_Lifeboat_Rules'] = 1 # Dick boat #3 pushed in 

# Emergency Boat 1, fourth boat lowered, only 12 people in this boat, strict 'women and children' rules not in force.
fd.loc[(fd.PassengerId==600),  'Lax_Lifeboat_Rules'] = 1 # Duff emergency lifeboat #1 run by Murdoch

# Boat 15, eighth boat lowered, most occupants were men, women and children not near this boat at the time.
fd.loc[(fd.PassengerId==932),  'Lax_Lifeboat_Rules'] = 1 # Karun boat #15 

# Boat jumpers
fd.loc[(fd.PassengerId==544),  'Jumped_into_Boat'] = 1 # Beane boat #9
fd.loc[(fd.PassengerId==1152), 'Jumped_into_Boat'] = 1 # de Messemaeker boat #15 helped row

# Men who went into the water and swam to boats.
fd.loc[(fd.PassengerId==225), 'Is_Swimmer'] = 1 # Hoyt
fd.loc[(fd.PassengerId==915), 'Is_Swimmer'] = 1 # Williams


# In[ ]:


Now we can make our predictions using the model.


# In[ ]:


#start with predictions
fd['Predict'] = -9
fd['Fate'] = ''
fd['Family_subclass'] = ''

# Start with all men dying. This covers all husbands dying and all father-child groups as well.
fd.loc[(fd.Title_c == 'man'),'Predict'] = 0
fd.loc[(fd.Title_c == 'man'),'Fate'] = 'Death, man'
printAccuracy()

# Boys die too.
fd.loc[(fd.Title_c == 'boy'),'Predict'] = 0
fd.loc[(fd.Title_c == 'boy'),'Fate'] = 'Death, boy'
printAccuracy()

# Women start by living, implying that married women live, and women in woman-parent groups also live
fd.loc[(fd.Title_c == 'woman'),'Predict'] = 1
fd.loc[(fd.Title_c == 'woman'),'Fate'] = 'Live, woman' 
printAccuracy()

# Set Boys in two person groups to live (100% match)
fd.loc[(fd.Title_c== 'boy') & (fd.Family_Count==2), 'Predict'] = 1
fd.loc[(fd.Title_c== 'boy') & (fd.Family_Count==2), 'Fate'] = 'Live, boy with parent'
printAccuracy()


# This gives us a 78.3% survival rate. Now classify the groups and provide explanation for how each person lived or died.

# In[ ]:




fd.loc[(fd.Lax_Lifeboat_Rules==1), 'Predict'] = 1
fd.loc[(fd.Lax_Lifeboat_Rules==1), 'Fate'] = 'Live, lax lifeboat rules'

fd.loc[(fd.Jumped_into_Boat==1), 'Predict'] = 1
fd.loc[(fd.Jumped_into_Boat==1), 'Fate'] = 'Live, jumped into boat'

fd.loc[(fd.Is_Swimmer==1), 'Predict'] = 1
fd.loc[(fd.Is_Swimmer==1), 'Fate'] = 'Live, swam to boat'

fd.loc[(fd.Was_Stubborn==1), 'Predict'] = 0
fd.loc[(fd.Was_Stubborn==1), 'Fate'] = 'Died, stubborn'

fd.loc[(fd.UnKnown_Cause==1), 'Predict'] = 0
fd.loc[(fd.UnKnown_Cause==1), 'Fate'] = 'Died, unknown cause'

fd.loc[(fd.Exposure==1), 'Predict'] = 0
fd.loc[(fd.Exposure==1), 'Fate'] = 'Died, exposure'

printAccuracy()


# Apply the eyewitness accounts raises to accuracy to 80.4%.  Now let's apply model the general model to the rest of the passengers.

# In[ ]:


#boys live if everyone in their group lives
fd.loc[(fd.Family_Count>2) & (fd.Title_c== 'boy') & (fd.Family_Pct==1), 'Predict'] = 1
printAccuracy()

#women die if eveyone in their group dies
fd.loc[(fd.Family_Count>2) & (fd.Title_c== 'woman') & (fd.Family_Pct==0), 'Predict'] = 0
printAccuracy()


# We end up with 85.1% accuracy.  
# 
# Additionally we have developed an excellent narrative of what actually transpired with the two person groups on that night as follows.
# 
# Titanic hits an iceberg and is sinking.  People start to assemble into groups, and these groups closely match the groups are are travelling with.  All two person groups follow three general rules very closely - men die, women live, and children with a parent live.  
# 
# Surprisingly, some men still live.  Many of the survivors can be characterized by having the ability to get into one of the first lifeboats leaving Titanic, meaning they were on deck early and in the vicinity of boats 3,5,7, or emergency boat 1.  These boats were being loaded by Murdoch, first officer of the Titanic, who was in charge on the vessel when the iceberg hit.  Murdoch was acutely aware of the gravity of the situation and was undoubtedly more concerned with saving lives, any lives, at the time than following a 'women and children first' rule.  As such the rule was not in force and many men - newlyweds, husbands, and dads - were allowed into these boats. A few other men surivived, some by taking risks (jumping into boats as they were lowered) and others by going down with the ship and swimming to the boats.
# 
# Unfortunately and surprisingly some women died who should have lived.  The vast majority of these women chose not leave their husbands and died with them onboard Titanic. There was also a tragedy and some cases that could not be explained. 
# 
# 
# In conclusion, we should identify some possibilities for feature engineering to extract these social, power, and authoritarian dynamics.  For men, there is a time and location element involved - being on deck early and in the right place (near boats 1,3,5 and 7) greatly improves your chances of survivial, at least for two person groups.  This information could be encoded in cabin number or deck information.  There is probably a correlation between party size and getting on deck quickly. Additionally, there may be some class or xenophobic information that could be encoded as well. 
# 
# For women, the challenge is to determine if the love they have for their husband is greater than their desire to live, and it would appear that this is very difficult to encode.  There is probably some corrlation to class (most of these women had second class tickets) and they are probably childless/not pregnant as well.
# 
# There are other possibiilties to improving models related to group size and adding a temporal element.  For example, after a man drops off a partner at a boat he is now a group of one, probably with different motivators and properties concerning his survival.  Additionally, it is clear that many of the rules change as the night continues - men are allowed on the first boats, then not allowed, then at the very end allowed again, etc.  
# 
# 
# 
# 
# 
# 
# 
# 
