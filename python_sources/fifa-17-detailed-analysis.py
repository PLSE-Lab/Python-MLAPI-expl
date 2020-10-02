#!/usr/bin/env python
# coding: utf-8

# ##**First Kaggle  - Detailed Analysis ** ##
# 
# ##Data Analysis
# ## Data Visualization  
# 
#  -  Top Player Analysis using *RadarChart *
#  -  Correlation using *HeatMap*
#  -  Attribute analysis using *count Plot*
#  -  *Seaborn* color palettes usage
#   
# ## Data Cleaning
# 
# ## Techniques used
#  - LinearRegression 
#  - Support Vector Machines
#  - SVR
#  - K Nearest Neighbours
# 
# 
# 

# # Header files

# In[1]:


import pandas as pd                                      # for reading data, exploring, cleaning, analysing and other reasons.
import numpy as np                                       # for mathematical fast calc. on data.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt                          #for plotting purposes
import seaborn as sns                                    # for aestheticalluy effective plotting 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time                                              # for time comparsion between different models

# this command is to print without writing 'print' statement of python!
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # Data Loading

# In[2]:


#Reading Data
df = pd.read_csv('../input/FullData.csv')                        # contains full data, to be analyzed
nat_namesDF = pd.read_csv('../input/NationalNames.csv')          # national Names dataset
club_namesDF = pd.read_csv('../input/ClubNames.csv')             # club names dataset
player_namesDF = pd.read_csv('../input/PlayerNames.csv')         # player names data set


# # Data exploring
# 

# ### 1) full data analysis
# ### 2) national names data analysis
# ### 3) club names data analysis
# ### 4) player names data analysis

# In[3]:


print(df.shape)                             #17588 rows, 53 cols.
print(df.head(5))                           
print(df.info())                             #for column name and their values data type


# In[4]:


print(pd.isnull(df).sum())                #checking null val. for each col 


# #### National_Kit , National_Position are filled with 16513 nan values maybe because many of the players have not been given a chance in National Team.
# ### Also 4 columns : 'Club_Position','Club_Kit','Club_Joining','Contract_Expiry' have 1 missing value each. So lets find out if, thats a common row, i.e. is it for 1 players who have all these attributes' value missing.  

# In[5]:


df_testing = df[['Club_Position','Club_Kit','Club_Joining','Contract_Expiry']].dropna(axis = 0,inplace = False,thresh = 1)    #use of thresh to get the idea
print(df.shape[0])
print(df_testing.shape[0])                           #its coming out to be less than original rows, i.e. assumption was right, only one player have these four attributes empty

#lets find out who is the player with these missing attributes, may be an unnown player.
inds = pd.isnull(df[['Club_Position','Club_Kit']]).any(axis = 1).nonzero()[0] 
print(inds)                                  # 383rd row of original data df has these attributes missing
empty_player = df.iloc[inds,:]
print(empty_player['Name'])                  #Didier Drogba: Chelsea Legend
didierDrogba = empty_player
print(pd.isnull(didierDrogba).sum())         #verifying our result about him, checking nan-valued attributes of him


# ### Didier Drogba :
# #### Chelsea Legend... scored Champions League 2012 final decider in penalty shootout aginst Bayern Munich!!!
# ##### so I will not remove this row and will fill the data properly.
# 

# In[6]:


df.loc[inds,'Club_Kit']=11.0                     #avoid chain indexing for avoiding copy = True
                                                 #alternativey with any condition. use: df.loc[df[<some_column_name>] == <condition>, <another_column_name>] = <value_to_add>
print(df.iloc[383,:])                            # I find that he is a free agent(not assigned to any club!, still gonna assign his old club, Chelsea)
df.loc[inds,'Club_Position'] = 'CF'
df.loc[inds,'Club'] = 'Chelsea'
df[['Club_Joining','Contract_Expiry']] =df[['Club_Joining','Contract_Expiry']].fillna(axis = 0,method='ffill',inplace = False)  #warning if inplace = True 
df.iloc[[383]]


# ## More analysis...

# In[7]:


df.describe(include=['O'])                 #string type data analysis only, by default include(), does not include string type columns!


# In[8]:


# to print all the columns effectively
pd.options.display.max_columns = 500
df.describe()                                    # integer type data


# ### from contract_expiry, max is 2023( but max. contract can be of 5 years) so we have to check these players.
# As such by looking stats, I cant much predict, i will use plotting for better analysis
# max. Rating is 94, one for Cristiano Ronaldo for sure
# 

# In[9]:


print((df.loc[:,'Contract_Expiry']==2023).sum())                 #894 entries, so I will believe in EA Sports for this behavior, and let it remain the same
print("no. of players having rating >=90 in FIFA 17:")
print((df.loc[:,'Rating'] > 89).sum())
print("Their Names: ")
bestRated_players = df[df.loc[:,'Rating']>89]#[['Name','Club','Rating']]    #Ronaldo is the winner as per the Ratings. 2 GKs are present too
bestRated_players


# # Analyzing the best rated players 
# ## Using Radar Chart 

# In[10]:


def _scale_data(datas,ranges): #data of 1 row is provided, with every column ranges
    
    (x1,x2) = ranges[0]
    diff = x2-x1
    fact = 0
    scaled_data = []
    for data, (y1,y2) in zip(datas,ranges):
        
        fact = ((data-y1)/(y2-y1))*diff
        scaled_data.append(fact+x1)
    return scaled_data

'''
class RadarChart
'''

class RadarChart():
    
    def __init__(self,fig,attributes,ranges,n_ordinate_levels=6):
        #n_ordinate_levels is for grid scaping, attributes being features, ranges is a list of all feature ranges
        angles = np.arange(0, 360, 360./len(attributes))
        
        
        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True,label="axis{}".format(i)) for i in range(len(attributes))] #we use label to distinguish b/w different axes, its a must, else we dont get different axes!
            
        _,text = axes[0].set_thetagrids(angles,labels = attributes)
        

        
        for txt,angle in zip(text,angles):
            txt.set_rotation(angle-90)
            txt.set_size(15)
            
 
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)  #patch is background
            ax.xaxis.set_visible(False)
            ax.grid("off")
        
        for i,ax in enumerate(axes):
            
            grid = np.linspace(*ranges[i],num = n_ordinate_levels)
            grid_label = [""] + [str(int(x)) for x in grid[1:]]                 
            ax.set_rgrids(grid,labels = grid_label,angle=angles[i]) ##rgrid is radial grid(circular)
                                                                    #grid here requires, starting circle distances, last circle dist, no. of circles i.e. *ranges[i],n_ordinate_levels
            ax.set_ylim(*ranges[i])
        
        self.angle = np.deg2rad(np.r_[angles,angles[0]])
        self.ranges  = ranges
        self.ax = axes[0]
        
    def plot(self,data,*args,**kwargs):                                    ##for plotting lines
        scaled_data = _scale_data(data,self.ranges)
        self.ax.plot(self.angle,np.r_[scaled_data,scaled_data[0]],*args,**kwargs)
            
    def fill(self,data,*args,**kwargs):
        scaled_data = _scale_data(data,self.ranges)                        ##for filling those lines generated polygon with color
        self.ax.fill(self.angle,np.r_[scaled_data,scaled_data[0]],*args,**kwargs)
        
    def legend(self,*args,**kwargs):                                       ##for labelling row name, i.e about self
        self.ax.legend(*args,**kwargs)
    
        
            
'''
class over 
'''


# In[11]:


attributes = ['Ball_Control','Dribbling','Marking','Aggression','Reactions', 'Attacking_Position',
       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',
       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',
       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing',
       'Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys',
       'GK_Positioning', 'GK_Diving', 'GK_Kicking', 'GK_Handling',
       'GK_Reflexes']
ranges = [[2**-20, df[attr].max()] for attr in attributes]
top_players = bestRated_players['Name'].unique().tolist()
datas = df[attributes].values
print(datas.shape)
print(type(datas))
print(type(top_players))
top_players
colors = sns.hls_palette(n_colors=len(top_players))
print(type(colors))


# In[12]:


fig = plt.figure(figsize=(25,25))
radar = RadarChart(fig,attributes,ranges)
for player,data,color in zip(top_players,datas,colors):
    radar.plot(data,color = color,label=player)  
    radar.legend(loc = 1, fontsize = 'large')
    radar.fill(data, alpha = 0.1, color = color)
plt.show()


# ### One can look for a given attribute, how players performance differ!
# ### Aslo you should reduce features and accordingly couple of other parameters to see only interested part of the above **RadarChart**!.... 

# ### Plotting RadarChart of CR7, Messi and Eden Hazard(my fav.)

# In[13]:


EdenHazard = df[df.loc[:,'Name']=='Eden Hazard'].T
EdenHazard


# In[14]:


colors = ['blue','green','red']             
chosen_players = ['Eden Hazard','Cristiano Ronaldo','Lionel Messi']


attributes = ['Ball_Control','Dribbling','Marking','Aggression','Reactions', 'Attacking_Position',
       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',
       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',
       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing',
       'Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys'
        ]
ranges = [[2**-20, df[attr].max()] for attr in attributes]
datas = df[attributes].values

fig = plt.figure(figsize=(25,25))
radar = RadarChart(fig,attributes,ranges)
for player,data,color in zip(chosen_players,datas,colors):
    radar.plot(data,color = color,label=player)  
    radar.legend(loc = 1, fontsize = 'large')
    radar.fill(data, alpha = 0.1, color = color)
plt.show()


# ### As you can see, *Messi's* Chart is almost inscribed inside other two's charts. Sorry Messi Fans (I am too!).
# ### Eden Hazard Chart's as can be seen, is comparable with *Cristiano Ronaldo*. Yayyyy! No wonder why Zinedine Zidane want *Hazard* to join Madrid!
# 

# ## Analysis using Heatmap

# ## * **Heatmap** is used to plot the correlation between different attributes*
# ### if corr ~ 1, then 2 attributes are highly correlated and follow the same trend
# ### if corr ~ -1, then 2 attributes are highly correlated and follow the inverse/opposite trend
# ### if corr ~ 0 then 2 attributes are nearly independent of each other

# In[15]:



def heatmap(df,figsize=(25,25),annot_size = 8,cmap=sns.cubehelix_palette(start = 0.2,rot = 0.3,dark = 0.15,light = 0.85,as_cmap = True)):
    corr = df.corr()
    _,ax = plt.subplots(1,1,figsize=figsize)
    sns.heatmap(corr,
               cbar=True,
               cbar_kws={'shrink':0.9},
               annot=True,
               annot_kws={'fontsize':annot_size},
               cmap = cmap
               )
    plt.show()                                        
   
heatmap(df)

    


# ## As you can see in above plot, I have used sns.cubehelix_palette() since I want the color change from light to dark!
# # in **seaborn**:
# ## for Diverging pattern use : *sns.diverging_palette*
# ## for Qualitative color palettes use: *sns.light_palette()* or *sns.color_palette()*
# ## for Sequential color palettes use: *sns.cubehelix_palette()*

# # *Miscellaneous Task*
# ## 3 other files analysis

# In[16]:


print(nat_namesDF.head())
print(nat_namesDF.shape)
print(nat_namesDF.describe(include=['O']))
print(pd.isnull(nat_namesDF).sum())                     # no null values


# In[17]:


print(club_namesDF.head())                            #633 clubs verifying from df 
print(df['Club'].unique().size)                       # 633 clubs, hence verified.
print(club_namesDF.shape)
print(club_namesDF.describe(include=['O'])) 
print(pd.isnull(club_namesDF).sum())


# In[18]:


print(player_namesDF.head())
print(player_namesDF.shape)
print(player_namesDF.describe(include=['O']))
print(pd.isnull(player_namesDF).sum())                    # no null values


# ###  by players_nameDF.describe(): out of 17588 names, only 17341 are uniques.

# # Data cleaning to test different models
# 

# In[19]:


dummy_df = df.copy()                  # dummy_df is the dataset on which i will work on... need to make a copy so as to not affect original dataset 


# In[20]:


dummy_df.keys()
dummy_df.shape
dummy_df.select_dtypes(['O']).shape        # 12 are string type cols.
dummy_df.select_dtypes([np.number]).shape  # 41 are int/float type columns


# In[21]:


string_columns = dummy_df.select_dtypes(['O']).columns           #string type column names
string_columns


# ### I will change height and weight to integers, remove BirthDate, since Age is already there

# In[22]:


dummy_df.Height.head()       #string have 'cm' attatched as a unit which need to be removed first
dummy_df.Weight.head()       #string have 'kg' attached as a unit which need to be removed first 


# In[23]:


dummy_df['Height'] = dummy_df.Height.str.replace('cm',' ')                        #replace cm with space
dummy_df['Height'] = dummy_df.Height.str.strip()                                   #strip any spaces
dummy_df['Height'] = pd.to_numeric(dummy_df['Height'], errors='coerce')            #srting to int64 type
dummy_df['Height'].dtype                                                          #check data type        


# In[24]:


dummy_df['Weight'] = dummy_df.Weight.str.replace('kg',' ')                        #replace cm with space
dummy_df['Weight'] = dummy_df.Weight.str.strip()                                   #strip any spaces
dummy_df['Weight'] = pd.to_numeric(dummy_df['Weight'], errors='coerce')            #srting to int64 type
dummy_df['Weight'].dtype                                                          #check data type    


# In[25]:


dummy_df.Height.head()
dummy_df.Weight.head()


# In[26]:


dummy_df.drop(['Birth_Date'],inplace = True,axis = 1)


# In[27]:


dummy_df.shape
string_columns =  dummy_df.select_dtypes(['O']).columns  # updating string col.


# ## I will encode these string valued columns, based on their unique values, and relevance
# ### First removing National_Kit and National_Position because of many Nan values
# 

# In[28]:


dummy_df.isnull().sum()


# In[29]:


dummy_df.drop(['National_Kit','National_Position'],inplace = True,axis = 1)
string_columns =  dummy_df.select_dtypes(['O']).columns  # updating string col.


# ### player's work rates (WR) are incredibly important in the way they behave on the pitch
# ### **WR = attacking/defensive**
# #### so i will split this attribute into two and assign weightage as per the work/rate in that department
# 

# In[30]:


dummy_df[['w_r_attack','w_r_defence']] = dummy_df.Work_Rate.str.split('/',expand=True)


# In[31]:


dummy_df.w_r_attack = dummy_df.w_r_attack.str.strip()
dummy_df.w_r_defence = dummy_df.w_r_defence.str.strip()


# In[32]:


dummy_df.w_r_defence = dummy_df.w_r_defence.map({'High':3,'Medium':2,'Low':1})
dummy_df.w_r_attack = dummy_df.w_r_attack.map({'High':3,'Medium':2,'Low':1})


# In[33]:



dummy_df.drop(['Work_Rate'],inplace = True,axis = 1)
string_columns = dummy_df.select_dtypes(['O']).columns     #updating string columns
string_columns                                             #'Name is not imp. for predecting', but I will keep it only for referencing
#dummy_df.drop(['Name'],axis = 1,inplace = True)
string_columns = dummy_df.select_dtypes(['O']).columns     #updating string columns


# ### Lets find out how w_r_attck and w_r_defence changes with age and for few selected top clubs!

# In[34]:


dummy_df['Club']


# In[35]:


chosen_clubs = ['FC Barcelona','Real Madrid','PSG','FC Bayern','Manchester Utd','Chelsea','Juventus']


# In[36]:


truth_table = dummy_df['Club'].apply(lambda x: x=='Real Madrid' or x=='FC Barcelona' or x =='PSG' or x=='FC Bayern' or x =='Manchester Utd' or x == 'Chelsea' or x == ' Juventus')
#Truth Table is for filtering out clubs I need only.


# In[37]:


plotting_df = dummy_df[truth_table]
plotting_df.head()


# In[38]:


fig,ax = plt.subplots(2,1,sharex = True)
fig.set_figheight(20)
fig.set_figwidth(20)

sns.barplot(x = 'w_r_attack',y = 'Age',hue = 'Club',data = plotting_df,ax = ax[0])
ax[0].set_xlabel('WORK Rate Attacking')
ax[0].set_ylim(15,40)                        #min age taking 15, and max. 40

sns.barplot(x = 'w_r_defence',y = 'Age',hue = 'Club',data = plotting_df,ax = ax[1])
ax[1].set_xlabel('WORK Rate Defensive')
ax[1].set_ylim(15,40)                        #min age taking 15, and max. 40


# As we can see,** No Barcelona Player's** work rate either defencive or attaking is 1 ... WowWwwWwww!

# In[39]:


string_columns.shape
string_columns


# ### **count plot **

# In[40]:


for column in string_columns:
    print("column: {} has these many unique values: {}".format(column,dummy_df[column].nunique()))
    if(dummy_df[column].nunique()<=30):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        ax = sns.countplot(x=column,hue='Skill_Moves',data = dummy_df,ax = ax)
        plt.show()
    else:
        print("column: {} has too many unique values to show on count plot.".format(column))


# mapping Right with 1 and Left with 0 in Preffered_Foot attribute

# In[41]:


dummy_df.Preffered_Foot = dummy_df.Preffered_Foot.map({'Right':1,'Left':0})


# In[42]:


string_columns = dummy_df.select_dtypes(['O']).columns
string_columns.shape
string_columns


# #### Nationality matters, so i will cluster nations into different areas, using basemaps later on( in next Notebook), 
# #### will delete Preffered_Position,Club_Joining, Label encode the rest

# ## First lets see how many players got  preferred position in their club!!!
# ### Making a new column, indicating this attribute which may affect their ratings, i.e. player whose preferred position is club position is likely to perform bettern than others

# In[43]:


(dummy_df['Preffered_Position'] == dummy_df['Club_Position']).sum()   #1891 players who play in club_position has their preffered_position
dummy_df['club_preferred_position_score'] = dummy_df['Preffered_Position'] == dummy_df['Club_Position']
dummy_df['club_preferred_position_score'].dtype                     #bool
dummy_df['club_preferred_position_score'] = dummy_df['club_preferred_position_score'].astype(int)
dummy_df['club_preferred_position_score'].shape


# In[44]:


dummy_df.drop(['Preffered_Position','Club_Joining'],axis = 1,inplace = True)


# In[45]:


string_columns = dummy_df.select_dtypes(['O']).columns         #updating
string_columns.shape                                           #will label encode all columns in it
string_columns


# In[46]:


string_columns[1:]


# In[47]:


le = LabelEncoder()
for column in string_columns[1:]:                                    #not encoding name, so as to use them for referencing later on
    dummy_df[column] = le.fit_transform(dummy_df[column])


# In[48]:


string_columns = dummy_df.select_dtypes(['O']).columns 
string_columns.shape                                           #finally 1, Phewww, not encoding Name


# # now since I have data only in numeric vals, i can test different models,
# ## My Target/Class/output will be :
# ### 1) Skill_Moves for Classification Models
# ### 2) Ratings for Regression Models
# # Aim: to predict skill moves and ratings for a given player

# In[49]:


output_df_classification  = dummy_df['Skill_Moves']
output_df_regression = dummy_df['Rating']


# In[50]:


dummy_df.drop(['Skill_Moves','Rating'],axis = 1,inplace = True)


# In[51]:


# checking regression and classification output correlation using heatmap

_,ax = plt.subplots(1,1)
outputDF = pd.DataFrame([output_df_classification,output_df_regression]).T
corr = outputDF.corr()
sns.heatmap(corr,
           ax = ax,
           cmap = sns.cubehelix_palette(start = 0,rot=0.1,as_cmap = True),
           annot = True,
            annot_kws={'fontsize':12},
            cbar = True
           )
plt.show()


# ### 0.25 score for correlation between Rating and Skill_Moves , i.e. not much dependent on each other
# ### because of positive correlation, Rating and Skill_Moves have a similar trend

# In[52]:


dummy_df.shape
output_df_classification.shape
output_df_regression.shape
### copying dummy_df to inputDF for model testing
inputDF = dummy_df.copy()


# ## reducing features

# ### as from the first heatmap, we saw high correlation between different GK attributes, I will only make their correlation map

# In[53]:


GK_attributes = df[['GK_Positioning','GK_Kicking','GK_Handling','GK_Reflexes']]
heatmap(GK_attributes,figsize=(10,10),annot_size = 12,cmap = sns.cubehelix_palette(start = -1,rot = 0.3,as_cmap = True))   


# ### since 5 attributes of GK are highly correlated with each other, I will use only one GK attribute,
# #### GK_Diving chosen for all

# In[54]:


inputDF.drop(['GK_Positioning','GK_Kicking','GK_Handling','GK_Reflexes'],inplace = True,axis = 1)


# In[55]:


heatmap(inputDF,cmap = sns.cubehelix_palette(start = 6,rot = 0.4,as_cmap=True))


# ### from more dense area for attribtes from Shot_Power to Volleys, plotting these attributes heatmap alone
# ### also Marking , Sliding Tackle, Standing_Tackle are highly correlated, so choosing Marking only
# ### similarly taking dribbling only, and dropping ball control

# In[56]:


inputDF.drop(['Ball_Control'],axis = 1,inplace = True)


# In[57]:


inputDF.drop(['Standing_Tackle','Sliding_Tackle'],inplace = True,axis = 1)


# In[58]:


heatmap(df[['Shot_Power','Finishing','Long_Shots','Curve','Freekick_Accuracy','Penalties','Volleys','GK_Diving']],figsize=(10,10),annot_size=12)


# ### Long_Shots and Shot_Power are highly correlated with correlance of 0.88, which is obvious, so only omitting one of these features

# In[59]:


inputDF.drop(['Long_Shots'],inplace = True,axis = 1)


# In[60]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,KFold


# In[61]:


x_trainR,x_testR,y_trainR,y_testR = train_test_split(inputDF,output_df_regression,random_state = 42)  #R for regression
train_names = x_trainR.iloc[:,0]
x_trainR = x_trainR.iloc[:,1:]
test_names = x_testR.iloc[:,0]
x_testR = x_testR.iloc[:,1:]
inputDF.shape


# In[62]:


model = LinearRegression()


# In[63]:


start = time.time()
model.fit(x_trainR,y_trainR)
print("fitting time : {}".format(time.time()-start))

start = time.time()
y_predR = model.predict(x_testR)
print("\nLinearRegression score is :")         # comes out to be 0.8456! that's really good because of just using simple LinearRegression
model.score(x_testR,y_testR)                   #Returns the coefficient of determination R^2 of the prediction.
print("testing time : {}".format(time.time() - start))


# In[64]:


y_predR.shape
y_testR.shape
np.isnan(y_predR).sum()
type(y_testR)
type(y_predR)


# In[65]:


comparisionRDF = pd.DataFrame(y_testR)
comparisionRDF['predicted'] = y_predR
comparisionRDF['player Name'] = test_names
comparisionRDF['error in Prediction'] = np.abs(comparisionRDF['predicted'] - comparisionRDF['Rating'])
comparisionRDF


# In[66]:


# Let's find out limit of error
print("Limit of error of our LinearR model is ({},{})".format(comparisionRDF['error in Prediction'].min(),comparisionRDF['error in Prediction'].max()))


# In[67]:


model.get_params()              
weights = model.coef_           #coefficients of all features/columns in inputDF.iloc[:,1:]
bias = model.intercept_         #constant term in a linear line equation
print("weights are:")
weights
print("Constant is :",bias)


# In[68]:


# cross_val_score result, reliable than simple score method!
crossScore = cross_val_score(model,X=inputDF.iloc[:,1:],y=output_df_regression,cv=KFold(n_splits = 5,shuffle = True,random_state = 42)).mean()
crossScore   


# ### Linear Regression found to be a good model to predict Rating with score of 0.816
# #### Now lets find out, if SVR is better than Linear Model to predict the same Regression problem
# 

# In[69]:


from sklearn.svm import LinearSVR,SVR


# In[70]:


model = LinearSVR(random_state = 42)


# In[71]:


start = time.time()
model.fit(x_trainR,y_trainR)
print("fitting time : {}".format(time.time()-start))

start = time.time()
y_predR = model.predict(x_testR)
score=model.score(x_testR,y_testR)
score
print("Testing time : {}".format(time.time()-start))


# ### LinearSVR, score is varying  too much, if I dont set random_state
# #### score is 0.8145 

# 
# <table>
#     <caption>Comparision of Different Regression Models</caption>
#     
#     <tr>
#         <th>Model Name</th>
#         <th>Fitting Time</th>
#         <th>Testing Time</th>
#         <th>Score</th>
#     </tr>
#     
#     <tr>
#         <td><b>LinearRegression</b></td>
#         <td>0.16456</td>
#         <td>0.08458</td>
#         <td><b>0.82105</b></td>
#     </tr>
#     
#     <tr>
#         <td><b>LinearSVR</b></td>
#         <td>2.65579</td>
#         <td>0.11985</td>
#         <td>0.81455</td>
#     </tr>
# 
# 
# </table>
# 

# ## Now time for Classification Task!

# In[72]:


x_trainC,x_testC,y_trainC,y_testC = train_test_split(inputDF,output_df_classification,random_state=42)
train_names = x_trainC.iloc[:,0]
x_trainC = x_trainC.iloc[:,1:]
test_names = x_testC.iloc[:,0]
x_testC = x_testC.iloc[:,1:]


# In[73]:


from sklearn.svm import SVC
from sklearn.decomposition import PCA  #PCA is helpful in reducing features/dimension without much loss of information!
from sklearn.preprocessing import StandardScaler


# # first scaling features, then applying
# ## 1). SupportVectorClassifier without PCA 

# In[74]:


sc = StandardScaler()
scaled_x_trainC = sc.fit_transform(x_trainC)
scaled_x_trainC = pd.DataFrame(scaled_x_trainC)
scaled_x_testC = sc.transform(x_testC)
scaled_x_testC = pd.DataFrame(scaled_x_testC)


# In[75]:


model = SVC(kernel='linear')
model.fit(scaled_x_trainC.iloc[:,:],y_trainC.iloc[:])
model.score(scaled_x_testC.iloc[:,:],y_testC)
model.support_.shape


# In[76]:


model = SVC(kernel = 'rbf')
model.fit(scaled_x_trainC,y_trainC)
model.score(scaled_x_testC,y_testC)
model.support_.shape


# ## 2). Support Vector Classifier with PCA

# In[77]:


pca = PCA(n_components = 2)


# In[78]:


pca_x_trainC = pca.fit_transform(scaled_x_trainC)
pca_x_testC = pca.transform(scaled_x_testC)
model = SVC(kernel = 'linear')
model.fit(pca_x_trainC,y_trainC)
model.score(pca_x_testC,y_testC)
model.support_.shape


# In[79]:


model = SVC(kernel='rbf')
model.fit(pca_x_trainC,y_trainC)
model.score(pca_x_testC,y_testC)
model.support_.shape


# ## Using K Nearest Neighbors i.e. KNN

# In[80]:


from sklearn.neighbors import KNeighborsClassifier


# In[81]:


start = time.time()
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_trainC,y_trainC)
print("fitting time: ",time.time()-start)
start = time.time()
print("score of KNN classification:")
model.score(x_testC,y_testC)
print("testing time: ",time.time()-start)


# 
# <table>
#     <caption>Comparision of Different Classification Models</caption>
#     
#     <tr>
#         <th>Model Name</th>
#         <th>Fitting Time</th>
#         <th>Testing Time</th>
#         <th>Score</th>
#     </tr>
#     
#     <tr>
#         <td><b>SVM without PCA</b></td>
#         <td>0.02657</td>
#         <td>0.00701</td>
#         <td>0.79281</td>
#     </tr>
#     
#     <tr>
#         <td><b>SVM with PCA</b></td>
#         <td>3.36240</td>
#         <td>0.00701</td>
#         <td>0.76438</td>
#     </tr>
# 
#      <tr>
#          <td><b>KNN</b></td>
#          <td>0.0441</td>
#          <td>0.7474</td>
#          <td>0.76256</td>
#       </tr>
# 
# 
# </table>

# ### Skill Moves as proved by SVM results is not much dependent on input features. Not just because of score but also due to high number of support vectors!
# ### High ratio of support vectors to total data points (rows of a dataset) usually indicates **overfitting**! 
# #### *PS*: note that in final comparision table,  fitting and testing time may change because of running again before publishing this kernel
# 

# ![](http://i.dailymail.co.uk/i/pix/2016/07/05/19/35FB51FB00000578-3675728-image-a-9_1467742012798.jpg)
# ## **Hope you guys liked it.**
# ## **Upvote and fork if you liked it.**
# ## Thanks for reading! Happy Kaggling.
# 

# In[ ]:




