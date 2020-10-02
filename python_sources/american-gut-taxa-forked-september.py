#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Welcome to this Carleton AI Society workshop!")


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'from IPython.display import Image\n\nimport os\nimport re\nimport numpy as np\nimport pandas as pd\nimport random\nimport warnings\n\nimport matplotlib\nimport matplotlib.pyplot as plt\nfrom matplotlib import collections as matcoll\nimport seaborn as sns\nimport lightgbm\n\nimport sklearn\nfrom sklearn import ensemble\nfrom sklearn import tree\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import *\nfrom sklearn.metrics import *\n\nfrom sklearn.metrics import roc_auc_score\nfrom scipy import stats')


# In[ ]:


warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = [15, 7.5]


# In[ ]:


L6_100nt = pd.read_csv('../input/L6_100nt.csv')


# In[ ]:


print("Finding columns that contain data about the participant's microbiota")
L6_pattern = re.compile("k__(\w*);p__(\w*);c__(\w*);o__(\w*);f__(\w*);g__(\w*)$")
L3_pattern = re.compile("k__(\w*);p__(\w*);c__(\w*);o__(\w*)$")
L2_pattern = re.compile("k__(\w*);p__(\w*)$")
L6_columns = [col for col in L6_100nt.columns if L6_pattern.match(col)]
L3_columns = [col for col in L6_100nt.columns if L3_pattern.match(col)]
L2_columns = [col for col in L6_100nt.columns if L2_pattern.match(col)]


# In[ ]:


def visualize_data(data, column, title, xAxis):
    """ Just a quick function to plot data easily """
    data[column] = pd.to_numeric(data[column], errors='coerce')
    fig, axes = plt.subplots(1, 2)
    female = data[data['SEX'] == 'female']
    male = data[data['SEX'] == 'male']
    fig.suptitle(title, fontsize=16)
    sns.distplot(male[column], bins=40, kde=False, ax=axes[0]);
    axes[0].set_ylabel('Number of Individuals (male)', fontsize=14)
    axes[0].set_xlabel(xAxis, fontsize=14)
    sns.distplot(female[column], bins=40, kde=False, ax=axes[1], color='r');
    axes[1].set_ylabel('Number of Individuals (female)', fontsize=14)
    axes[1].set_xlabel(xAxis, fontsize=14)
    return fig

def filter_data(study):
    """ Removes unwanted rows or modify them to limit the space of the task """
    study = L6_100nt[L6_100nt['STUDY'] == study]

    study['BMI_CORRECTED'] = study['BMI_CORRECTED'].replace("no_data",np.nan).replace("Unspecified",np.nan).replace("Unknown",np.nan).astype(float)
    study['AGE_CORRECTED'] = study['AGE_CORRECTED'].replace("Unspecified",np.nan).replace("Unknown",np.nan).astype(float)    
    study = study[(study['AGE_CORRECTED'].isnull()) | (study['AGE_CORRECTED'] >= 18)]
    
    subset_underweight = study[(study['BMI_CORRECTED'] < 18.5)]
    subset_healthyweight = study[(study['BMI_CORRECTED'] >= 18.5) & (study['BMI_CORRECTED'] < 25)]
    subset_overweight = study[(study['BMI_CORRECTED'] >= 25) & (study['BMI_CORRECTED'] < 30)]
    subset_obese = study[(study['BMI_CORRECTED'] >= 30) & (study['BMI_CORRECTED'])]
    
    study = pd.concat([subset_underweight, subset_healthyweight, subset_overweight, subset_obese])
    
    #Label Smoothing
    study['SUBSET_UNDERWEIGHT'] = (study['BMI_CORRECTED'] < 18.5).astype(float) * 0.8
    study['SUBSET_HEALTHYWEIGHT'] = ((study['BMI_CORRECTED'] >= 18.5) & (study['BMI_CORRECTED'] < 25)).astype(float) * 0.8
    study['SUBSET_OVERWEIGHT'] = ((study['BMI_CORRECTED'] >= 25) & (study['BMI_CORRECTED'] < 30)).astype(float) * 0.8
    study['SUBSET_OBESE'] = (study['BMI_CORRECTED'] >= 30).astype(float) * 0.8
    return study


# In[ ]:


print("Filtering based on the study, as many scientific studies were involved")
meta_study = pd.concat([filter_data(study) for study in L6_100nt['STUDY'].unique()])
meta_study = meta_study[~meta_study['#SampleID'].duplicated()]


# In[ ]:


print("Just a quick overview of the height of the participants. This can be used as a simple sanity check.")
sns.set()
plt.show(visualize_data(meta_study, 'HEIGHT_CM', 'Height Distribution', 'Height (cm)'))
#plt.show(visualize_data(meta_study, 'WEIGHT_KG', 'Weight Distribution', 'Weight (kg)'))
#plt.show(visualize_data(meta_study, 'BMI_CORRECTED', 'BMI Distribution', 'BMI'))
#plt.show(visualize_data(meta_study, 'AGE_CORRECTED', 'Age Distribution', 'Age (years)'))


# In[ ]:


#Grab every participant ignoring BMI class sizes 
subset_underweight = meta_study[meta_study['SUBSET_UNDERWEIGHT'].astype(bool)]
subset_healthyweight = meta_study[meta_study['SUBSET_HEALTHYWEIGHT'].astype(bool)]#.sample(n=1000, random_state=SEED)
subset_overweight = meta_study[meta_study['SUBSET_OVERWEIGHT'].astype(bool)]#.sample(n=391, random_state=SEED)
subset_obese = meta_study[meta_study['SUBSET_OBESE'].astype(bool)]#.sample(n=391, random_state=SEED)


# In[ ]:


meta = pd.concat([subset_underweight, subset_healthyweight, subset_overweight, subset_obese])
meta = meta.fillna(0)


# In[ ]:


# Filtering data to only consider one source of microbiota in the participant's body 
# Filtering further to ignore participants who recently used antibiotic
data = meta[meta['BODY_SITE'] == 'UBERON:feces']
data = data[data['SUBSET_ANTIBIOTIC_HISTORY'] | (data['ANTIBIOTIC_HISTORY'] == 'Year') | (data['ANTIBIOTIC_HISTORY'] == '6 months')]


# In[ ]:


features = meta[L6_columns].var().sort_values(ascending=False).index[:600].tolist()


# In[ ]:


data = data.groupby(["HOST_SUBJECT_ID"]).first()

under = data[data["SUBSET_UNDERWEIGHT"] == 0.8]
over = data[data["SUBSET_OBESE"] == 0.8]

sample_size = min(under.shape[0], over.shape[0])

under = under.sample(sample_size)
over = over.sample(sample_size)

obesity = pd.concat([under, over])

obesity["obesity_target"] = (obesity["SUBSET_OBESE"] == 0.8)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(obesity[features], obesity["obesity_target"], test_size=0.30)


# In[ ]:


#model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
trained_model = model.fit(x_train, y_train)
predictions = model.predict(x_test)
cm = confusion_matrix(y_test, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(roc_auc_score(y_test, predictions))
pd.DataFrame(data=cm)


# In[ ]:


dimensions = 300
n_points = 1000

cursed_data = np.random.normal(0, 1, size=(n_points, dimensions))
cursed_label = cursed_data[:,0] > 0


# In[ ]:


plt.scatter(cursed_data[:,0], y=[0]*(n_points), c=cursed_label, cmap="Accent")


# In[ ]:


plt.scatter(cursed_data[:,0], cursed_data[:,1], c=cursed_label, cmap="Accent")


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cursed_data[:,0], cursed_data[:,2], cursed_data[:,1], c=cursed_label, cmap="Accent")


# In[ ]:


cursed_features = range(0,100)
cursed_df = pd.DataFrame(cursed_data)
cursed_df["target"] = cursed_label
x_train, x_test, y_train, y_test = train_test_split(cursed_df[cursed_features], cursed_df["target"], test_size=0.50)


# In[ ]:


model = sklearn.neighbors.KNeighborsClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[ ]:


model = sklearn.tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(obesity[features], obesity["obesity_target"], test_size=0.30)
model = sklearn.tree.DecisionTreeClassifier()
trained_model = model.fit(x_train, y_train)
predictions = model.predict(x_test)
cm = confusion_matrix(y_test, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(roc_auc_score(y_test, predictions))
pd.DataFrame(data=cm)


# In[ ]:


trials = []
rf_trials = []
for trial in range(100):
    x_train, x_test, y_train, y_test = train_test_split(obesity[features], obesity["obesity_target"], test_size=0.30)
    model = sklearn.tree.DecisionTreeClassifier()
    trained_model = model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = (roc_auc_score(y_test, predictions))
    trials.append(score)
    
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=20)
    trained_model = model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = (roc_auc_score(y_test, predictions))
    rf_trials.append(score)
    


# In[ ]:


h = plt.hist(trials,bins=10, alpha=0.5, label=f"Single Decision Tree: {np.mean(trials)}")
h = plt.hist(rf_trials,bins=10, alpha=0.5, label=f"Random Forest: {np.mean(rf_trials)}")
plt.legend(loc='upper left')


# In[ ]:


plant_group = data.groupby(["HOST_SUBJECT_ID"]).first()

planty = plant_group[plant_group["TYPES_OF_PLANTS"] == "More than 30"]
not_planty = pd.concat([plant_group[plant_group["TYPES_OF_PLANTS"] == "6 to 10"], plant_group[plant_group["TYPES_OF_PLANTS"] == "Less than 5"]])

sample_size = min(planty.shape[0],not_planty.shape[0])

planty = planty.sample(sample_size)
not_planty = not_planty.sample(sample_size)

plant_set = pd.concat([planty,not_planty])
targets = [1]*len(planty)+[0]*len(not_planty)
plant_set["target"] = targets
plant_set = plant_set.sample(frac=1.0)

x_train, x_test, y_train, y_test = train_test_split(plant_set[features], plant_set["target"], test_size=0.20)

model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
trained_model = model.fit(x_train, y_train)
predictions = model.predict(x_test)
cm = confusion_matrix(y_test, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("AUC score:", roc_auc_score(y_test, predictions))
pd.DataFrame(data=cm)


# In[ ]:


sns.countplot(y="TYPES_OF_PLANTS", hue="obesity_target", data=obesity)


# In[ ]:


print(list(obesity.columns))


# In[ ]:


import umap


# In[ ]:


reducer = umap.UMAP()
embedding = reducer.fit_transform(obesity[features])
colors = sklearn.preprocessing.LabelEncoder().fit_transform(obesity["ALCOHOL_FREQUENCY"])
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in colors])


# In[ ]:


reducer = sklearn.decomposition.PCA()
embedding = reducer.fit_transform(obesity[features])
colors = sklearn.preprocessing.LabelEncoder().fit_transform(obesity["obesity_target"])
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in colors])


# In[ ]:


reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(obesity[features], y=obesity["obesity_target"])
colors = sklearn.preprocessing.LabelEncoder().fit_transform(obesity["obesity_target"])
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in colors])


# In[ ]:


embedding = reducer.transform(obesity[features])
colors = sklearn.preprocessing.LabelEncoder().fit_transform(obesity["obesity_target"])
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in colors])


# In[ ]:


#https://msystems.asm.org/content/3/3/e00031-18


# In[ ]:


reducer


# In[ ]:




