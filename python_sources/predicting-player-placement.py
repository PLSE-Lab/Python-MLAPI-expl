#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ## Objective
# 
# 
# Starcraft II is an incomplete information multiplayer real-time strategy game that centers around gathering resources to build units and defeat an opponent. Players of this game generally play in an online ladder that places players into different leagues based on their performance against other players using a ranking system similar to ELO in chess. Unlike in chess, players must race against their opponent to execute their strategy, while at the same time gathering information on their opponent and defending their attacks.
# 
# When joining the current ladder system, players are required to play 10 placement matches in order to find their leagues. However, this tends to under estimate their skills and takes many games for players to arrive at the correct placement. During this adjustment period, it could give other players who are correctly placed bad experiences by being completely outmatched.
# 
# The objective of this project is to predict the league placement of a Starcraft II player by only using the information contained inside a replay. The real-time nature of the game allows us to differentiate players through the speed and efficiency of their in game actions. This is in an attepmt to  reduce the time taken to arrive at the correct placement.
# 
# This report is organised as follows:
# 
# * [Section 2 (Overview)](#overview) outlines our methodology. 
# 
# * [Section 3 (Simple Preprocessing)](#data_prepro) makes sure the data is clean and what we expect.
# 
# * [Section 4 (Data Exploration)](#explore) explores and visualizes the data. 
# 
# * [Section 5 (Data Preparation)](#data_prep) summarizes the data preparation process and our model evaluation strategy.
# 
# * [Section 6 (Hyperparameter Tuning)](#param_tuning) describes the hyperparameter tuning process for each classification algorithm. 
# 
# * [Section 7 (Performance Comparison)](#comparison) presents model performance comparison results.
# 
# * [Section 8 (Limitations)](#limitations) discusses a limitations of our approach and possible solutions. 
# 
# * [Section 9 (Summary)](#summary) provides a brief summary of our work in this project.
# 
# 
# Compiled from Jupyter Notebook, this report contains both narratives and the Python codes used for data processing, model buiding and evaluation.
# 
# 
# # Overview <a name="overview"></a>
# 
# ## Target Feature
# 
# The target feature is the league index which is numbers 1-8 representing Bronze, Silver, Gold, Platinum, Diamond, Master, Grandmaster, Professional leagues. However, it was found that we have very small proportions of Bronze, Grandmaster and Professional league players in the dataset. In order to prevent this imbalance from impacting the predictive performance of our models, these leagues will be rolled into the leagues closest to them, resulting in the following index:
# 
# 1. Bronze-Silver
# 
# 2. Gold
# 
# 3. Platinum
# 
# 4. Diamond
# 
# 5. Master-Grandmaster-Professional
# 
# 
# The objective being to accurately place a player into one of these leagues just by using data that can be automatically mined from a replay.
# 
# ## Packages Used

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
# For plot marker colours
import colorlover as cl
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn import metrics


# ## Methodology
# 
# We buill build the following classifiers to predict the target feature:
# 
# * K-Nearest Neighbours (KNN)
# 
# * Decision Tree (DT)
# 
# * Naive Bayes (NB)
# 
# First, the dataset from phase 1 will be transformed to suit our analysis. This includes reencoding the target feature to balance out its populations, as well as dropping unnecessary features and scaling the descriptive features. 
# 
# The data will be randomly split with stratificationinto training and test sets with a 70:30 ratio. This results in a training set with 2336 rows and a test set with 1002 rows. Stratification is necessary to ensure that each validation set has the same proportion of target classes as in the original dataset because the population is imbalanced.
# 
# As part of the model fitting pipeline, we will be using ANOVA F-value, Mutual Information and Chi-squared methods to select the top 3, 4, 5, 6, 7, 10 or 15 (full set) of features.
# 
# We will conduct a 5-fold repeated stratified cross-validation (with 3 repeats) to fine-tune hyperparameters of each classifier using Accuracy as the performance metric. This will be done in a single pipeline along with feature selection for each model with parallel processing using "-2" cores. 
# 
# Once the best model of the three has been identified, using hyperparameter search on the training data, we conduct a 10-fold repeated (3 repeats) cross-validation on the test data and perform a paired t-test to see if any performance difference is statistically significant. In addition, we compare the classifiers with respect to their precision scores, recall scores and confusion matrices on the test data.

# # Simple Preprocessing <a name="data_prepro"></a>
# 
# ## Data Import

# In[ ]:


sc = pd.read_csv("/kaggle/input/skillcraft/SkillCraft.csv")
sc.head()


# ## Data Cleaning
# 
# For this step, the feature types are checked against the documentation.

# In[ ]:


print(f"Dimension of the data set is{sc.shape}\n")
print(f"Data Types are:")
print(sc.dtypes)


# Because the goal of this project is to make predictions based on the data mined from submitted replays, the Age, HoursPerWeek and TotalHours columns are not important and will not be a feature used in predictions. Thus I shall remove these columns:

# In[ ]:


sc.drop(columns=["Age", "HoursPerWeek", "TotalHours"], inplace=True)


# I shall also perform a surface level check for NaN values.

# In[ ]:


print("Number of missing value for each feature:")
print(sc.isnull().sum())


# Because all the data in the table currently has numerical values, I do a quick summary to check for abnormalities.

# In[ ]:


sc.describe()


# From the above summary, the values for the data seem reasonable. 

# # Data Exploration <a name="explore"></a>
# 
# For convenience and consistency, I defined a few variables the ensure the correct league labels and colours. The league labels will be used instead of LeagueIndex for the visualizations. Additionally, functions for creating for univariate box plot paired with histogram as well as multivariate violin plots was created.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Name the leagues
league_lbls = ["Bronze","Silver","Gold","Platinum","Diamond","Master","Grandmaster"]
league_indexs = sc["LeagueIndex"].unique()
league_indexs.sort()
league_lbls_dict = dict()
for i, ind in enumerate(league_indexs):
    league_lbls_dict[ind] = league_lbls[i]
league_labeled = sc["LeagueIndex"].replace(league_lbls_dict)


def clrgb_to_hex(rgb):
    rgb = re.search("\(([^\)]+)\)", rgb).group(1).split(",")
    hex_clr = "#"
    for n in rgb:
        val = hex(int(n))[2:]
        if len(val)<2:
            val = "0"+val
        hex_clr+=val
    return hex_clr


# Define league colours for consistency
league_colours_raw = cl.scales['8']['qual']['Paired']
league_colours = []
for i, clr in enumerate(league_colours_raw):
    league_colours.append(clrgb_to_hex(league_colours_raw[i]))

league_colours_dict = dict()
for i, lbl in enumerate(league_lbls):
    league_colours_dict[lbl] = league_colours[i]
    
    
def box_hist_plot(x, title, w, h):
    fig, (ax_box, ax_hist)= plt.subplots(2, sharex=True,gridspec_kw={"height_ratios": (.15,.85)})
    fig.set_size_inches(w, h)
    
    ax_box.set_xlim(0,x.max())
    ax_hist.set_xlim(0,x.max())
    
    sns.boxplot(x, ax=ax_box)
    sns.distplot(x, ax=ax_hist)
    ax_box.set(yticks=[])
    
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    ax_box.set_title(title)
    ax_hist.set_title(None)
    plt.show()
    
def violin_plot(y, title, w, h):
    plt.figure(figsize=(w, h))
    ax1 = sns.violinplot(x=league_labeled, y=y, palette=league_colours_dict, order=league_lbls)
    ax1.set_ylim(0,)
    ax1.set(xlabel='League')
    plt.title(title)
    plt.show()
    
def auto_plot(feature, fig_num):
    box_hist_plot(sc[feature], f"Figure {fig_num}: {feature} Distribution", 11, 8)
    violin_plot(sc[feature], f"Figure {fig_num+1}: {feature} by League", 11, 8)


# ## Target Feature Distribution
# 
# 
# Around the time when this data was collected(2011), the targeted ratio of ladder players were as follows:
# 
# * Grandmaster Top 200 players in a region
# 
# * Master 2%
# 
# * Diamond 18%
# 
# * Platinum 20%
# 
# * Gold 20%
# 
# * Silver 20%
# 
# * Bronze 20%
# 
# 
# Generally professional players are also Grandmasters in their respective regions, however there is a wide enough skill gap between Grandmasters who are and aren't professionals that the distinction matters.

# In[ ]:


def league_dist():
    global fig_count
    
    #labels
    lab = league_lbls
    #values: counts for each category
    val = sc["LeagueIndex"].value_counts().sort_index().values.tolist()
    pct = [x/sum(val)for x in val]
    
    fig1, ax1 = plt.subplots()
    #ax1.pie(val, labels=lab, autopct='%1.2f%%', pctdistance=0.8,shadow=True, startangle=90)
    
    wedges, texts = ax1.pie(pct, wedgeprops=dict(width=0.5), startangle=90, colors=league_colours)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax1.annotate(f"{val[i]} {league_lbls[i]} ({pct[i]*100:.2f}%)", xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)
    
    ax1.set_title("Figure 1: Distribution of Leagues", y=1.2)
    
league_dist()


# Because the data was collected through online gaming communities and social media, it is not surprising that the proportion of data available for Bronze and Silver league games are very small compared to the ladder distribution at the time, whereas Master league games exceed its ladder proportion by a large margin. Players in higher leagues tend to be more involved in the community, keeping up with the latest strategies and discussions. On the other hand, lower league players are likely new or casual players, logging in just to play the game for fun without necessarily engaging with the community outside of the game client. It also appears that this dataset does not contain professional players.
# 
# 
# ## Data Visualization
# 
# 
# All of the features we have are numerical, the auto_plot function is applied to all of them. This function produces a plot of the feature on its own, as well as a plot split by the target feature.
# 
# 
# ### APM

# In[ ]:


auto_plot("APM",2)


# In figure 2, the mean of Action Per Minutes (APM) is about 100, which is what I would expect from an average player. The positive skew generally comes from 2 sources:
# 
# 1. Good players being able to perform much more effective actions than the average player.
# 
# 2. A player repeatedly performing the same actions in quick succession that do not contribute to the outcome of the game.
# 
# 
# Point number 2 can be a source of noise which lowers the reliability of this feature. However the proportion of players that would play like this is likely to be small as it takes a lot of effort while potentially negatively impacting the players ability to play the game.
# 
# 
# In figure 3, there is an upward trend in APM as the league increases. However, there is fair bit of overlap between the leagues.
# 
# 
# ### SelectByHotkeys

# In[ ]:


auto_plot("SelectByHotkeys",4)


# In figure 4, a large proportion of players don't use selection by hotkeys very much, as it is likely more intuitive to select units on the screen using a mouse.
# 
# 
# As games get more competitive, efficiency in actions become more important. Selecting by hotkeys while requiring continous maintenance throughout the game, is much faster and allows the player to not only select units anywhere on the map, but also allows instantaneous movement of the player camera to show their selected units. Unsurprisingly, hotkey usage goes up as the league improves, with a relatively large spike in professional games in figure 5. 
# 
# 
# ### AssignToHotkeys

# In[ ]:


auto_plot("AssignToHotkeys",6)


# Figures 6 and 7 show that the assignments of hotkeys are generally more balanced between the leagues, however this also tells us that proportionally many people are aware of hotkeys, and go through the setup process in game, but then do not use actually use them as much for selection.
# 
# 
# ### UniqueHotkeys

# In[ ]:


auto_plot("UniqueHotkeys",8)


# With a maximum of 10 assignable hotkeys, figure 8 shows a fairly reasonable distribution of unique hotkey use with a mean of 4. Using more than 4 or 5 unique hotkeys takes practice and muscle memory as the movements require the player to move their hand across the keyboard using the number keys 1 to 0 without looking down in combination with other modifier keys to maximize efficiency.
# 
# 
# There is actually an interesting step up of 1 unique hotkey used per timestamp per every 2 leagues in figure 9. But this also makes sense remembering the discussion in the section on SelectByHotkeys.
# 
# 
# ### MinimapAttacks, MinimapRightClicks

# In[ ]:


auto_plot("MinimapAttacks",10)
auto_plot("MinimapRightClicks",12)


# Figure 11 shows that minimap attacks are very few and far between in a game, and in a lot of cases players don't even use it. There are many ways to perform attacks and the usage of the minimap can be a stylistic choice for a lot of players. However, we do still see a trend upward in use as the league improves, as players multitask more and improve the efficiency of their actions by using all the tools available to them.
# 
# The logic behind this MinimapRightClicks is similar to that of MinimapAttacks, however this is used more in general likely because the action is easier to perform.
# 
# 
# ### NumberOfPACs, GapBetweenPACs, ActionLatency, ActionsInPAC

# In[ ]:


auto_plot("NumberOfPACs",14)
auto_plot("GapBetweenPACs",16)
auto_plot("ActionLatency",18)
auto_plot("ActionsInPAC",20)


# These 3 features essentially show us the performance values for each stage in a PAC. 
# 
# 
# Similar to other activities which require fast reaction time, the more highly ranked players perceive events and take action more quickly. NumberOfPACs, GapBetweenPACs and ActionLatency improve significantly as the league improves. These feature essentially quantifies the players ability to multitask. ActionsInPAC don't change very much between leagues. When NumberOfPACs are taken into account, ActionsInPAC don't matter as much as long as players have more PACs and those actions are efficiently used.
# 
# 
# ### TotalMapExplored

# In[ ]:


auto_plot("TotalMapExplored",22)


# This feature does not seem to provide much information regarding the target feature. This is also quite unreliable as the amount of the map that is explored by a player can vary wildly depending on the design of the map, was well as the type of strategy the player employs. 
# 
# 
# ### WorkersMade

# In[ ]:


auto_plot("WorkersMade",24)


# This feature also does not seem to provide much information regarding the target feature. It is heavily dependent on type of strategies the both players employs as well as game length. There is a balance that must be struck between using resources to create workers or improving the players army, often once a certain number of workers are created, no more will be made for the rest of the game. It can be argued that this can be used to differentiate very weak players from average or better players, but I doubt it will be very helpful in the context of this notebook.
# 
# 
# ### UniqueUnitsMade, ComplexUnitsMade, ComplexAbilitiesUsed

# In[ ]:


auto_plot("UniqueUnitsMade",26)
auto_plot("ComplexUnitsMade",28)
auto_plot("ComplexAbilitiesUsed",30)


# Similar to WorkersMade, it will be very hard to differentiate players by using these features. 
# 
# Depending on the strategy, a player can just win by making as little as 1 unit type, often having a very lean, specialized army can make greater impact than a large one with a lot of different units. Complex units as well as their abilities are accessible to any player with enough time and resources, but could still be used to differentiate weak players from average or better players.

# # Data Preparation <a name="data_prep"></a>
# 
# ## Reencoding the Target Feature
# 
# As discussed in the overview, the target feature in the dataset has some fairly imbalanced populations. To remedy this, first we will separate our target variable from our dataset and merge them with they larger neighbours. Below shows the current population of each league in our dataset. Below shows the current population of each league in our dataset.

# In[ ]:


leagues = sc.LeagueIndex
sc_data = sc.drop(columns='LeagueIndex')
leagues.value_counts()


# Thus we will condense the number of classifications in the target feature from 8 to 5 to balance out the populations.

# In[ ]:


# Reencode
leagues.replace({2:1, 3:2, 4:3, 5:4, 6:5, 7:5, 8:5}, inplace=True)
# New Value Counts
leagues.value_counts()


# ## Normalizing Features
# 
# Because the data contains values from very different ranges, it is important to make sure our features are in the same scale.

# In[ ]:


def normalize_data(data):
    scaler = preprocessing.MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    
    # When the data has been transformed, a np.array is returned,
    # So we have to convert it back to a dataframe, and insert column names
    return pd.DataFrame(data_norm, columns=data.columns)
    
sc_norm = normalize_data(sc_data)
sc_norm.describe()


# ## Feature Selection and Ranking
# 
# Here we will use ANOVA F-value, Mutual Information and Chi-squared scoring on the full dataset to rank and select features. 

# In[ ]:


def plot_scores():
    def get_k_best(data, target, method, k):
        skb = SelectKBest(method, k = k)
        skb.fit(data.values, target.values)
        fs_indices = np.argsort(skb.scores_)[::-1]

        return pd.DataFrame({"features": data.columns[fs_indices].values, 
                      "scores": skb.scores_[fs_indices]})
    
    fig, axs = plt.subplots(ncols=3, figsize=(20,6))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    
    titles = []
    data = []
    titles.append("ANOVA F-value")
    data.append(get_k_best(sc_norm, leagues, f_classif, len(sc_norm.columns)))
    titles.append("Mutual Information")
    data.append(get_k_best(sc_norm, leagues, mutual_info_classif, len(sc_norm.columns)))
    titles.append("Chi-squared")
    data.append(get_k_best(sc_norm, leagues, chi2, len(sc_norm.columns)))
    
    for i in range(3):
        p = sns.barplot(x='features', y='scores', data=data[i], ax=axs[i])
        p.set_xticklabels(p.get_xticklabels(), rotation=90)
        p.set_title(titles[i])
        

plot_scores()


# We can see from the above plot that the top 5 spots are generally taken up by Action Latency, APM, NumberOfPACs, SelectByHotkeys and GapBetweenPACs. The fetures that are ranked lower tend to move around depending on the selection method, which will make a difference when not all features are used to train the model.

# ## Train-Test Splitting
# 
# Because this dataset only has 3395 rows, there is no need for sampling. However, it is still necessary to split the dataset into train and test partitions. The partitions will be created using stratification at a 70:30 train to test ratio.

# In[ ]:


sc_train, sc_test, leagues_train, leagues_test = train_test_split(sc_norm, leagues, 
                                                                  test_size = 0.3, random_state=1,
                                                                  stratify = leagues)

print(f"Training dataset shape: {sc_train.shape}")
print(f"Test dataset shape: {sc_test.shape}")
print(f"Training target shape: {leagues_train.shape}")
print(f"Test target shape: {leagues_test.shape}")


# ## Model Evaluation Strategy
# 
# Our model will be trained and tuned on 2336 rows of data, and tested on 102 rows of data as seen in the previous section.
# 
# For each model, we will use 5-fold repeated stratified cross-validation evaluation method (with 3 repeats) for hyperparameter tuning.

# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)


# # Hyperparameter Tuning <a name="param_tuning"></a>
# 
# ## K-Nearest Neighbours (KNN)
# 
# Here we will stack feature selection and grid search for hyperparameter tuning (via cross-validation) in a "pipeline". The same Pipieline methodology will be used for NB and DT. Our pipeline will include multiple selectors, which were discussed in the Feature Selection and Ranking section.
# 
# KNN hyperparameters include:
# 
# * The number of neighbours, where we will initially consider n of 150, 160 ,170 ,180 ,190, and 200.
# 
# * The distance metric p, where we will initially consider p values of 1 (Manhattan), 2 (Euclidean), and 5 (Minkowski)

# In[ ]:


def run_KNN_pipe(n_neighbours, p):
    pipe_KNN = Pipeline([('selector', SelectKBest()), 
                         ('knn', KNeighborsClassifier())])

    params_pipe_KNN = {'selector__score_func': [f_classif, mutual_info_classif, chi2],
                       'selector__k': [3, 4, 5, 6, 7, 10, sc_norm.shape[1]],
                       'knn__n_neighbors': n_neighbours,
                       'knn__p': p}

    gs_pipe_KNN = GridSearchCV(estimator=pipe_KNN, 
                               param_grid=params_pipe_KNN, 
                               cv=cv,
                               n_jobs = -1,
                               scoring='accuracy',
                               verbose=0)

    gs_pipe_KNN.fit(sc_train, leagues_train);
    
    return gs_pipe_KNN


gs_pipe_KNN = run_KNN_pipe([150, 160 ,170 ,180 ,190, 200], [1,2,5])


# In[ ]:


gs_pipe_KNN.best_params_


# In[ ]:


gs_pipe_KNN.best_score_


# The top 7 features were selected by the ANOVA F-value test and is used by our KNN model to find its 170 nearest neighbours at a p of 1. The output shows that this optimal KNN model has a mean accuracy score of 0.4408. 
# 
# To better gauge how much and why this optimal set of hyperparameters performs better compared its peers, the results should be formatted into a more readable state. Below is a function provided by the teaching staff of the RMIT course for Machine Learning:

# In[ ]:


# custom function to format the search results as a Pandas data frame
def get_search_results(gs):

    def model_result(scores, params):
        scores = {'mean_score': np.mean(scores),
             'std_score': np.std(scores),
             'min_score': np.min(scores),
             'max_score': np.max(scores)}
        return pd.Series({**params,**scores})

    models = []
    scores = []

    for i in range(gs.n_splits_):
        key = f"split{i}_test_score"
        r = gs.cv_results_[key]        
        scores.append(r.reshape(-1,1))

    all_scores = np.hstack(scores)
    for p, s in zip(gs.cv_results_['params'], all_scores):
        models.append((model_result(s, p)))

    pipe_results = pd.concat(models, axis=1).T.sort_values(['mean_score'], ascending=False)

    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']
    columns = columns_first + [c for c in pipe_results.columns if c not in columns_first]

    return pipe_results[columns]

results_KNN = get_search_results(gs_pipe_KNN)
results_KNN.head(5)


# We will plot these expected results to get an understanding of how changing the hyperparameters affect the prediction accuracy.

# In[ ]:


def plot_KNN_results(res):
    def get_selector_data(d, p, sel):
        return d[(d.knn__p==p) & (d.selector__score_func==sel)].iloc[:,[0,4,6]]
    
    rows = len(res["knn__p"].unique())
    
    fig, axs = plt.subplots(ncols=3, nrows=rows, figsize=(20,rows*6), sharey='all')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    titles = []
    data = []
    temp = res.copy().infer_objects()
    for i, p in enumerate(res["knn__p"].unique()):
        titles.append(f"ANOVA F-value, p={p}")
        data.append(get_selector_data(temp, p, f_classif))
        titles.append(f"Mutual Information, p={p}")
        data.append(get_selector_data(temp, p, mutual_info_classif))
        titles.append(f"Chi-squared, p={p}")
        data.append(get_selector_data(temp, p, chi2))
    
    row = 0
    col = 0
    ax = None
    for i in range(rows*3):
        if col%3==0 and col!=0:
            col=0
            row+=1
        if rows == 1:
            ax = axs[col]
        else:
            ax=axs[row,col]
        p = sns.lineplot(x='knn__n_neighbors', 
                         y='mean_score', 
                         hue='selector__k', 
                         data=data[i], 
                         ax=ax, 
                         palette=sns.color_palette("Set1", 7))
        p.set_title(titles[i])
        p.legend(loc='lower right')
        col+=1
        
plot_KNN_results(results_KNN)


# ### KNN Fine Tuning
# 
# We can see in the graphs that many of the lines are trending upward as we increase our n value, so we will rerun the pipeline with higher values to compare. Because p=1 is the best performing value for p in general, we will stick with that for this new run.

# In[ ]:


gs_pipe_KNN2 = run_KNN_pipe([210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350], [1])


# In[ ]:


gs_pipe_KNN2.best_params_


# In[ ]:


gs_pipe_KNN2.best_score_


# In[ ]:


results_KNN2 = get_search_results(gs_pipe_KNN2)
results_KNN2.head(5)


# In[ ]:


plot_KNN_results(results_KNN2)


# This time around, the top 10 features were selected by the Mutual Info (and ANOVA F-value) method and is used by our KNN model to find its 240 nearest neighbours at a p of 1 with a mean accuracy score of 0.4443. 
# 
# This is a minor improvement when compared with the best mean score of 0.4408 from the last run. The graphs above show that many of the lines have either levelled out or begun to decline.

# ## Decision Trees (DT)
# 
# Here we will build a decision tree that uses the Gini index to split the data. We will use GridSearchCV to determine the optimal values for the maximum depth and minimum sample split.
# 
# DT hyperparameters include:
# 
# * The maximum depth of the DT, where we will initially consider the values 5, 7 and 9.
# 
# * The minimum number of samples required to be at a leaf node, where we will initially consider the values 2, 3, 5, 7, 9, 11.

# In[ ]:


def run_dt_pipe(max_depth, min_split):
    pipe_DT = Pipeline([('selector', SelectKBest()), 
                         ('dt', DecisionTreeClassifier(criterion='gini'))])

    params_pipe_DT = {'selector__score_func': [f_classif, mutual_info_classif, chi2],
                       'selector__k': [3, 4, 5, 6, 7, 10, sc_norm.shape[1]],
                       'dt__max_depth': max_depth,
                       'dt__min_samples_split': min_split}
 
    gs_pipe_DT = GridSearchCV(estimator=pipe_DT, 
                               param_grid=params_pipe_DT, 
                               cv=cv,
                               n_jobs = -1,
                               scoring='accuracy',
                               verbose=0)

    gs_pipe_DT.fit(sc_train, leagues_train);
    
    return gs_pipe_DT


gs_pipe_DT = run_dt_pipe([5, 7, 9], [2, 3, 5, 7, 9, 11])


# In[ ]:


gs_pipe_DT.best_params_


# In[ ]:


gs_pipe_DT.best_score_


# The top 7 features were selected by the Chi-squared test and is used with a maximum depth of 5 and minimum split value of 2. The output shows that this optimal DT model has a mean accuracy score of 0.4174. 

# In[ ]:


results_DT = get_search_results(gs_pipe_DT)
results_DT.head(5)


# As we did before, we will plot these expected results to get an understanding of how changing the hyperparameters affect the prediction accuracy.

# In[ ]:


def plot_DT_results(res):
    def get_selector_data(d, p, sel):
        return d[(d.dt__max_depth==p) & (d.selector__score_func==sel)].iloc[:,[0,5,6]]
    
    rows = len(res["dt__max_depth"].unique())
    
    fig, axs = plt.subplots(ncols=3, nrows=rows, figsize=(20,rows*6), sharey='all')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    titles = []
    data = []
    temp = res.copy().infer_objects()
    for i, p in enumerate(res["dt__max_depth"].unique()):
        titles.append(f"ANOVA F-value, max_depth={p}")
        data.append(get_selector_data(temp, p, f_classif))
        titles.append(f"Mutual Information, max_depth={p}")
        data.append(get_selector_data(temp, p, mutual_info_classif))
        titles.append(f"Chi-squared, max_depth={p}")
        data.append(get_selector_data(temp, p, chi2))
    
    row = 0
    col = 0
    ax = None
    for i in range(rows*3):
        if col%3==0 and col!=0:
            col=0
            row+=1
        if rows == 1:
            ax = axs[col]
        else:
            ax=axs[row,col]
        p = sns.lineplot(x='dt__min_samples_split', 
                         y='mean_score', 
                         hue='selector__k', 
                         data=data[i], 
                         ax=ax, 
                         palette=sns.color_palette("Set1", 7))
        p.set_title(titles[i])
        p.legend(loc='lower right')
        col+=1

        
plot_DT_results(results_DT)


# ## Fine Tuning DT
# 
# There is a very clear downward trend for the mean score as the max depth increases. Thus the next step would be to rerun the model with depths that are less than 5. After that, we will also increase the values that are being checked for min_samples_split to check for better values.

# In[ ]:


def compare_depths():
    def get_selector_data(d, p, sel):
        return d[(d.dt__min_samples_split==p) & (d.selector__score_func==sel)].iloc[:,[0,4,6]]
    
    pipe_DT = Pipeline([('selector', SelectKBest()), 
                         ('dt', DecisionTreeClassifier(criterion='gini'))])

    params_pipe_DT = {'selector__score_func': [f_classif],
                       'selector__k': [3, 4, 5],
                       'dt__max_depth': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                       'dt__min_samples_split': [2]}
 
    gs_pipe_DT = GridSearchCV(estimator=pipe_DT, 
                               param_grid=params_pipe_DT, 
                               cv=cv,
                               n_jobs = -1,
                               scoring='accuracy',
                               verbose=0)

    res = get_search_results(gs_pipe_DT.fit(sc_train, leagues_train));
    res = res.infer_objects()
    p = sns.lineplot(x='dt__max_depth', 
                     y='mean_score', 
                     hue='selector__k', 
                     data=get_selector_data(res,2,f_classif), 
                     palette=sns.color_palette("Set1", 3))
    #p.set_title(titles[i])
    p.legend(loc='lower right')
    
compare_depths()


# We can see that the value 5 for maximum depth is clearly the optimal one.
# 
# Next, we will try many different values for minimum sample splitting threshold. We are able to do so many as locking down the depth frees up the computing time.

# In[ ]:


gs_pipe_DT2 = run_dt_pipe([5], [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201])


# In[ ]:


gs_pipe_DT2.best_params_


# In[ ]:


gs_pipe_DT2.best_score_


# In[ ]:


results_DT2 = get_search_results(gs_pipe_DT2)
results_DT2.head(5)


# In[ ]:


plot_DT_results(results_DT2)


# Similar to before, the top 3 features were selected by the Chi-squared test and is used in conjunction with max_depth=5 and min_samples_split=181 to obtain a mean accuracy score of 0.4187. 
# 
# This is a minor improvement when compared with the best mean score of 0.4174 from the last run. The graphs above show that it is unlikely we will find a better value by increasing the value of min_samples_split as there is a steep decline at around 161.

# ## Naive Bayes (NB)
# 
# Here we will build a Gaussian Naive Bayes model and optimize the value of var_smoothing. We will perform a logspace search for the optimal var_smoothing, starting with the 100 to $10^{-2}$ with a total of 100 values. There is no prior information available to provide this model.
# 
# Each descriptive feature must follow a Gaussian distribution in a NB model, a power transformation is first performed on the input data.
# 
# Because the Chi-sqaured test does not work with negative values, it will be excluded from the pipeline of this section.

# In[ ]:


np.random.seed(1)

def run_NB_pipe(var_smoothing):
    pipe_NB = Pipeline([('selector', SelectKBest()), 
                         ('nb', GaussianNB())])

    params_pipe_NB = {'selector__score_func': [f_classif, mutual_info_classif],
                       'selector__k': [3, 4, 5, 6, 7, 10, sc_norm.shape[1]],
                       'nb__var_smoothing': var_smoothing}

    gs_pipe_NB = GridSearchCV(estimator=pipe_NB, 
                               param_grid=params_pipe_NB, 
                               cv=cv,
                               n_jobs = -1,
                               scoring='accuracy',
                               verbose=0)
    
    sc_train_transformed = PowerTransformer().fit_transform(sc_train)
    gs_pipe_NB.fit(sc_train_transformed, leagues_train);
    
    return gs_pipe_NB


gs_pipe_NB = run_NB_pipe(np.logspace(2,-2, num=100))


# In[ ]:


gs_pipe_NB.best_params_


# In[ ]:


gs_pipe_NB.best_score_


# The top 10 features were selected by the Mutual Info selector and using var_smoothing=0.16297508346206435, our model obtains a mean score of 0.4603.

# In[ ]:


results_NB = get_search_results(gs_pipe_NB)
results_NB.head(5)


# In[ ]:


def plot_NB_results(res):
    def get_selector_data(d, sel):
        return d[(d.selector__score_func==sel)].iloc[:,[0,4,5]]
    
    fig, axs = plt.subplots(ncols=2, figsize=(20,6), sharey='all')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    
    titles = []
    data = []
    temp = res.copy().infer_objects()
    
    titles.append(f"ANOVA F-value")
    data.append(get_selector_data(temp, f_classif))
    titles.append(f"Mutual Information")
    data.append(get_selector_data(temp, mutual_info_classif))
    
    for i in range(2):
        p = sns.lineplot(x='nb__var_smoothing', 
                         y='mean_score', 
                         hue='selector__k', 
                         data=data[i], 
                         ax=axs[i], 
                         palette=sns.color_palette("Set1", 7))
        p.set_xscale("log")
        p.set_title(titles[i])
        p.legend(loc='lower right')


plot_NB_results(results_NB)


# As there doesn't seem to be any pattern pointing to values of var_smoothing improving the mean score, there won't be any fine tuning done for this model.

# # Performance Comparison <a name="comparison"></a>
# 
# Now that we have created our optimized models for each classifier, we will use them on the test data with 10 fold repeated cross-validation. Due to the randomness of the cross-validation process, any differences in performance of the optimized models will be put through pairwise t-tests to determine if they are statistically significant.
# 
# First, we will calculate the cross-validation scores for each of our models.

# In[ ]:


cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2)

cv_results_KNN = cross_val_score(estimator=gs_pipe_KNN2.best_estimator_,
                                 X=sc_test,
                                 y=leagues_test, 
                                 cv=cv2, 
                                 n_jobs=-1,
                                 scoring='accuracy')
cv_results_KNN.mean()


# In[ ]:


cv_results_DT = cross_val_score(estimator=gs_pipe_DT2.best_estimator_,
                                X=sc_test,
                                y=leagues_test, 
                                cv=cv2, 
                                n_jobs=-1,
                                scoring='accuracy')
cv_results_DT.mean()


# In[ ]:


sc_test_transformed = PowerTransformer().fit_transform(sc_test)

cv_results_NB = cross_val_score(estimator=gs_pipe_NB.best_estimator_,
                                X=sc_test_transformed,
                                y=leagues_test, 
                                cv=cv2, 
                                n_jobs=-1,
                                scoring='accuracy')
cv_results_NB.mean()


# Second, we conduct a paired t-test for the accuracy score between the following model combinations:
# 
# * KNN vs. DT
# 
# * KNN vs. NB
# 
# * DT vs. NB

# In[ ]:


print(stats.ttest_rel(cv_results_KNN, cv_results_DT))
print(stats.ttest_rel(cv_results_KNN, cv_results_NB))
print(stats.ttest_rel(cv_results_DT, cv_results_NB))


# Looking at these results, we conclude that at a 95% significance level, NB is statistically the best model in terms of accuracy when compared on the test data.
# 
# We will now compare the models using other measures such as:
# 
# * Precision
# 
# * Recall
#     
# * F1 Score (the harmonic average of precision and recall)
# 
# * Confusion Matrix

# In[ ]:


pred_KNN = gs_pipe_KNN.predict(sc_test)

pred_DT = gs_pipe_DT2.predict(sc_test)

sc_test_transformed = PowerTransformer().fit_transform(sc_test)
pred_NB = gs_pipe_NB.predict(sc_test_transformed)

print("\nK-Nearest Neighbour Report") 
print(metrics.classification_report(leagues_test, pred_KNN))
print("\nDecision Tree Report") 
print(metrics.classification_report(leagues_test, pred_DT))
print("\nNaive Bayes Report") 
print(metrics.classification_report(leagues_test, pred_NB))


# In[ ]:


print("\nConfusion matrix for K-Nearest Neighbour") 
print(metrics.confusion_matrix(leagues_test, pred_KNN))
print("\nConfusion matrix for Decision Tree") 
print(metrics.confusion_matrix(leagues_test, pred_DT))
print("\nConfusion matrix for Naive Bayes") 
print(metrics.confusion_matrix(leagues_test, pred_NB))


# From the reports and confusion matrix, NB and KNN are very closely matched for the highest averages for precision, recall and accuracy. However, upon closer inspection NB and KNN has a very wide range in all its scores such as going from a recall of 0.61 in class 1 to 0.14 in class 2, whereas DT is the most balanced. The confusion matrix does align with these findings.
# 
# Given this information, I would say that DT could possibly perform the best in practice.

# # Limitations and Proposed Solutions<a name="limitations"></a>
# 
# Our dataset currently only has 3395 rows of data, which was when it was collected, a very small proportion of the existing player base. This also resulted in the imbalance in the spread of leagues in the data that caused us to merge several leagues together. Additionally, these are gathered from voluntarily submitted replays, which could make the model biased towards being accurate in predicting placements for active members of the community, who likely have more strategic knowledge, rather than the general population. These could possibly be solved by increasing the number of submitted replays from the community. Professional replays can be obtained when tournaments release their replay packs. We can also try to work with the developer Blizzard Entertainment to allow all players to opt in to automatic replay submissions for analysis after a ladder match.
# 
# The features found in the data alone are likely not sufficient to very accurately predict a player's league placement. They are very focused on the physical interactions the player has with the game, on top of that, there is a fair bit of overlap between the leagues. A core aspect of the game is resource management for which there is a very clear difference between skilled and unskilled players as well. Additional features such as average income, average savings, spending habits (such as use of production queueing) and production building activity counters could help paint a more well rounded picture of the players in game.
# 
# When creating the models, we chose to use pipelines to try multiple feature selectors and many hyperparameter combinations, and then refining the model after. While it seems to have been effective to a degree, we could potentially improve our predictive power by using in depth analysis to manually select features and hyperparameter values.

# # Summary<a name="summary"></a>
# 
# The F-value selector selected the 7 best features which were used to create a Decision Tree (DT) model that obtained the highest cross-validated Accuracy score on the training data. Although on average the DT model model on average does not outperform the Naive Bayes and k-Nearest Neighbours (KNN) models, it is the most balanced model with the smallest range for all of its classification measures.
# 
# With a mean training score of 0.4187 and 0.4034 on the test data, the model's performance is not good. However, it does provide better chance of success than purely guessing placement in 5 categories.
