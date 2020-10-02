#!/usr/bin/env python
# coding: utf-8

# # Covid(RAN) - COVID-19 Research & Analytics Notebook
# 
# <h2> COVID-19 : Transmission, Incubation, and Environmental Stability </h1>
# <h2> Making conclusions form data and validating with Literary References </h2>

# <h2> Notebook details </h2>
# 
# 1. The dataset mentioned underneath is uploaded from the kaggle COVID-19 Dataset. (https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset).
# 
# 2. Apart Python Notebooks, visualizations are also imported from Tableau, PowerBI and analysis is also done on Excel and SAS.
# 
# 3. The details for the same is added to this python notebook wherever applicable.
# 
# 4. The Literary references for this notebook was fetched from the https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge dataset.
# 
# 5. The above mentioned data was wrangled to serve its best purpose.
# 
# 6. Due to the complexity of the notebook some lines of code are under comments. They can be uncommented.
# 
# <h2> Dataset Description </h2>
# 
# 'In response to the COVID-19 pandemic, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 29,000 scholarly articles, including over 13,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.' - Kaggle
# 
# <h2> Importing the Essential Libraries </h2>

# In[ ]:


import pandas as pd                 #Using to read the csv files
import matplotlib.pyplot as plt     #For data plotting
import numpy as np                  #Building mathematical calculations
import glob                         #Retrieve the file/path name
import re                           #Regular-Expression to fetch data from text
import json                         #Reading the JSON Documents
import string                       #For doing string operations

#Essential ML Libraries

from sklearn.feature_extraction.text import HashingVectorizer      #Performing Hashing Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer        #Tfidf vector for text analysis
from sklearn.model_selection import train_test_split               #For train-test-split for the dataset
from sklearn.ensemble import RandomForestClassifier                #Random-forest-classifier
from sklearn.neighbors import KNeighborsClassifier                 #KNN Classifier
from sklearn import svm                                            #Support Vector Machine classifier
from sklearn import metrics                                        #Module to check the model accuracy

#Essential Neural Network Libraries

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD

#Other essential libraries
import plotly.express as px                                        #Plotly for plotting the COVID-19 Spread.
import plotly.offline as py                                        

print("Mentioned Libraries Successfully Imported")


# <h1> Importing the Literature Dataset </h1>

# In[ ]:


"""literature_data = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
literature_data.head(2)"""


# <h2> Wrangling the above dataset </h2>

# In[ ]:


"""literature_data['doi'].astype('str') #Changing datatype of the column doi to string
literature_data.dtypes"""


# <h1> Manipulations with the JSON Documents </h1>

# In[ ]:


"""#Getting the filepaths of JSON Documents with GLOB
filepaths_json = glob.glob('../input/CORD-19-research-challenge//**/*.json', recursive=True)  #Setting Recursive Feature as True.

#printing the datatype for the filepaths_json variable
type(filepaths_json)

#Printing the first two elements from the list
filepaths_json[0:2]"""


# <H2> Fetching data from the JSON Document </H2>
# 
# (Special Thanks to maksimeren for the code to fetch the details.)
# (maksimeren Notebook available at - https://www.kaggle.com/maksimeren/covid-19-literature-clustering)
# 

# In[ ]:


"""class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
        
            self.body_text = []
            # Code to read the abstract of JSON File
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Code to read the body of the JSON File
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(filepaths_json[0])
print(first_row)

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data

dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(filepaths_json):
    if idx % (len(filepaths_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(filepaths_json)}')
    content = FileReader(entry)
    
    # get the information from the JSON File
    meta_data = literature_data.loc[literature_data['sha'] == content.paper_id]
    # If no information is found for the provided paper, the paper is skipped.
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = literature_data.loc[literature_data['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(". ".join(authors[:2]) + "...")
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
literature_data = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
literature_data.head()"""


# <h2> Understanding the dataset Above </h2>
# 
# The dataset above highlights the respective paperid along with the abstract, body text, the title of the paper, its authors and a short summary of the abstract containing the first 300 chracters from the abstract column
# 
# <h2> Wrangling the Generated dataset </h2>

# In[ ]:


"""#Checking for the null values

literature_data.isnull().sum()"""


# <h2> Findings from the Null value operation </h2>
# 
# 1. We find 0 null values for the columns paper_id, abstract, body_text.
# 2. However some null values are present in the author and title column which has to be taken care of.
# 
# We first drop the duplicate values in the prior column and then check the null value again.

# In[ ]:


"""#Dropping the duplicate values.
literature_data.drop_duplicates(['abstract', 'body_text'], inplace=True)

#Checking the count of null values again.
literature_data.isnull().sum()

#Dropping the null values.
literature_data.dropna(inplace=True)

#Removing Punctuations
literature_data['body_text'] = literature_data['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

#Converting to Lower case
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

literature_data['body_text'] = literature_data['body_text'].apply(lambda x: lower_case(x))"""


# <h2> Implementing Machine Learning on the previous dataset </h2>
# 
# The steps involved in this procedure is as under:
# 
# 1. The column body_text is taken as the main value to form the segmentation algorithm. (Taken as X)
# 2. Using Tfidf Vectorization method the text was vectorized.
# 3. The body_text using the fit method was then fitted to the vectorization instance.
# 4. Using mini-batch kmeans the data was fitted and labels was added at the end of the dataset.

# <h2> Please Note </h2>
# 
# The following code runs for 26,000+ rows in the dataset hence the computatiion time is higher. It could take anywhere between a couple of hours to run the underlying code below hence the code has been marked as a markdown. The code could safely run and give the same values. 
# 
# An already executed version of this code, executed on Rapids.ai Tesla P100 GPU has been added as a public dataset on Kaggle by me. The same is used here to do more analysis. The dataset is made available publically on kaggle under the url - https://www.kaggle.com/aestheteaman01/covid19-literary-analysis-dataset-covlad. The same is imported in this notebook.

# <h2> The code for vectorization and K-Means is under </h2>
# 
# Runtime for this code is higher as already discussed.

# <h3> #Selecting the Body_Text form the dataset and storing the value in the variable X. </h3>
# 
# X = literature_covid['body_text']
# 
# 
# <h3> #Implementing the Tf-IDF-Vectorizer to vectorize the dataset. </h3>
# 
# from sklearn.feature_extraction.text import TfidfVectorizer
# 
# vectorizer = TfidfVectorizer(max_features=2**12)
# 
# X = vectorizer.fit_transform(df_covid['body_text'])

# <h3> #Implementing K-Means Clustering Algorithm to the generated Sparse Matrix. </h3>
# 
# from sklearn.cluster import KMeans
# 
# k = 17
# 
# kmeans = MiniBatchKMeans(n_clusters=k)
# 
# y_pred = kmeans.fit_predict(X)

# <h2> Loading the Literary Clustered Dataset </h2>
# 
# The followind dataset is generated from the code written in  markdown above.

# In[ ]:


literature_data = pd.read_csv('../input/covid19-literary-analysis-dataset-covlad/COVID-Literature-Analysis.csv')
literature_data.head(2)


# <h2> Basic Data Wrangling for the generated Dataset </h2>

# In[ ]:


#Dropping off the unecessary rows.
literature_data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1, inplace=True)

#Genrating a column total that reads the word count of the column body_text
literature_data['Word_Count'] = literature_data['body_text'].apply(lambda x: len(x.strip().split()))

#Viewing the dataset
literature_data.head(2)


# <h1> Implementing a Classification Code over the labeled dataset</h1>

# In[ ]:


X = literature_data['body_text']  #Selecting the independent variable for classification

#Building a Hashing Vectorizer to convert the text to vectorized numbers via a sparse matrix
from sklearn.feature_extraction.text import HashingVectorizer

# hash vectorizer instance
vectors = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)

# features matrix X
X = vectors.fit_transform(X)


# In[ ]:


#Selecting the column for labels as the independent variable (Labels from K-Means Clustering)

Y = literature_data['Labels']


# In[ ]:


#Implementing a train test split over the dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=4)


# <h2> Implementing the ML Algorithms </h2>
# 
# <h2> Random Forest Regressor </h2>

# In[ ]:


#Implementing a Random Forest Classifier for the dataset.

from sklearn.ensemble import RandomForestClassifier

rnd = RandomForestClassifier()
rnd.fit(x_train,y_train)
yhat = rnd.predict(x_test)


# In[ ]:



#Searching for the model accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,yhat)


# <h2> Support Vector Machine Classifier </h2>

# In[ ]:


#Implementing a SVM Classifier

from sklearn import svm
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(x_train,y_train)

yhat = clf.predict(x_test)


# In[ ]:


#Searching for the model accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,yhat)


# <h2> Implementing KNN Classifier to the dataset </h2>

# In[ ]:


#Implementing a  for KNN Classifier to the dataset.

from sklearn.neighbors import KNeighborsClassifier

k = 9 #Initializing the value of K for classification

neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski',p=2).fit(x_train,y_train)
yhat = neigh.predict(x_test)


# In[ ]:


#Searching for the model accuracy

from sklearn import metrics
metrics.accuracy_score(y_test,yhat)


# <h2> Hunting for a neural network for higher accuracy </h2>
# 
# We develop a neural network to check if the accuracy goes up for the classification for the model

# In[ ]:


#Importing the essential libraries for neural network development
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD

#Initiating the model
model = Sequential()

#Adding layers to the model
model.add(Dense(2000, activation='sigmoid', input_shape=(4096,)))
model.add(Dense(750, activation='sigmoid')) 
model.add(Dense(17, activation='softmax'))

#Model Compilation and fitting the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=512, epochs=200, validation_data=(x_test,y_test))

#Neural Network Evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print(score[0])
print(score[1])


# <h2> Analyzing the classification model </h2>
# 
# 1. Since the neural network didin't provided a substantial increase in the overall accuracy for the model, the basic ML Models are taken into considerations for the prediction purposes.
# 
# 2. The Random Forest Classifier which possessed the higher accuracy, is taken as a initiator to help with the classification example. 
# 
# 3. Further, the neural network can also be developed to attain a much higher accuracy.

# <h2> Prediction with the Random Forest Classifier Model </h2>
# 
# 1. We develop a function which takes input as the text and predicts the label. 
# 2. The more is the detail for the text inputed, the higher is the chances for a better prediction.
# 3. The code returns a label which then can be used to find the related articles from the main COVID Literature dataset.

# In[ ]:


def prediction_text():
    print("Enter the text to search for")  #Input statement to enter the text to search for.
    
    #Basic Editing for the input text
    
    text_search = str(input())
    text_search = text_search.lower()                            #Converts to lower case.
    text_search = re.sub('[^a-zA-z0-9\s]','',text_search)        #Removes punctuations
    text_search = [text_search]   
    
    
    #Using Hash Vectorizer to analyze the input text and convert it into a sparse matrix.
    hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)   
    test_sample = hvec.fit_transform(text_search)
    
    #Predicting the results
    pred = rnd.predict(test_sample)
    pred = pred[0]
    
    print("The matched Label for the input query is = {}".format(pred))
    print("\nReturning the dataset (named - search_results) for the label")
    
    label_dataset = literature_data['Labels'] == pred        #Returning the dataset that matches with the prediction label
    search_results = literature_data[label_dataset]
    
    print(search_results.head(4))


# Note: The function call prediction_text() is used to envoke the above function. The more detail is the input the better is the model accuracy for prediction the corect label.

# In[ ]:


#Using the above function
prediction_text()


# <h1> Importing the COVID-19 Global Cases Dataset </h1>

# In[ ]:


covid_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_data.head()


# <h1> Basic Data Wrangling </h1>
# 
# 1. Deleting the not required column - (Last update)
# 2. Resetting the index with the column SNo.
# 3. Checking for missing and null values

# In[ ]:


covid_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_data.head()


# In[ ]:


#Dropping of the column Last Update
covid_data.drop('Last Update', axis=1, inplace=True)

#Resetting the index to SNo Column
covid_data.set_index(['SNo'], inplace=True)

#Viewing the dataset
covid_data.head()

#Replacing NaN Values in Province/State with a string "Not Reported"
covid_data['Province/State'].replace(np.nan, "Not Reported", inplace=True)

#Printing the dataset
covid_data.head()


# <h2> Creating an interactive Map for COVID-19 Cases </h2>

# In[ ]:


#Creating the interactive map
py.init_notebook_mode(connected=True)

#GroupingBy the dataset for the map
formated_gdf = covid_data.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['Date'] = pd.to_datetime(formated_gdf['ObservationDate'])
formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

#Plotting the figure
fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], 
                     projection="natural earth", animation_frame="Date", 
                     title='Spread of COVID-19 Virus')

#Showing the figure
fig.update(layout_coloraxis_showscale=False)
py.offline.iplot(fig)


# <h2> Grouping the countries together </h2>

# In[ ]:


#Groping the same cities and countries together along with their successive dates.

country_list = covid_data['Country/Region'].unique()

country_grouped_covid = covid_data[0:1]

for country in country_list:
    test_data = covid_data['Country/Region'] == country   
    test_data = covid_data[test_data]
    country_grouped_covid = pd.concat([country_grouped_covid, test_data], axis=0)
    
country_grouped_covid.reset_index(drop=True)
country_grouped_covid.head()


# Now the dataset generated above is Wrangled and is ready for observations. We check once to ensure the graphs of confirmed cases are plotted.
# 
# For this case we select to plot the graph for Hubei Province of China, to see total confirmed cases over time. The following is done via the MatPlotLib Library.
# 
# The underlying code below prints the confirmed cases vs observation date for the city that is provided as the input

# In[ ]:


def plot_case_graph():
    
    print("Enter the city to be searched\n")
    search_city = str(input())

    #Draws the plot for the searched city

    search_data = country_grouped_covid['Province/State'] == search_city       #Selecting the city
    search_data = country_grouped_covid[search_data]                           #Filtering the dataset

    x = search_data['ObservationDate']
    y = search_data['Confirmed']
    b = search_data['Confirmed'].values
    
    
    a = b.shape   
    a = a[0]
    growth_rate = []    
    
    for i in range(1,a):                                       #Loop to calculate the daily growth rate of cases
        daily_growth_rate = ((b[i]/b[i-1])-1)*100
        growth_rate.append(daily_growth_rate)                                      

    growth_rate.append(daily_growth_rate)
        
    data = {'Growth' : growth_rate}
    b = pd.DataFrame(data)
    
    #Plotting the chart for confirmed cases vs date     
        
    plt.figure(figsize=(15,5))
    plt.bar(x,y,color="#9ACD32")                              
    plt.xticks(rotation=90)
    
    plt.title('Confirmed Cases Over time in {}'.format(search_city))
    plt.xlabel('Time')
    plt.ylabel('Confirmed Cases')

    plt.tight_layout()
    plt.show()
    
    #Plotting the chart daily growth rate in confirmed COVID-19 Cases.
    
    plt.figure(figsize=(15,5))
    plt.plot(x,b,color='red', marker='o', linestyle='dashed',linewidth=2, markersize=8,label="Daily Growth Rate of New Confirmed Cases")
    plt.xticks(rotation=90)
    
    plt.title('Confirmed Cases Over time in {}'.format(search_city))
    plt.xlabel('Time')
    plt.ylabel('Percentage Daily Increase')

    plt.tight_layout()
    plt.show()
    
plot_case_graph()


# <h1> Task 1 - Seasonality of COVID-19 Transmission </h1>

# We first observe the COVID-19 Spreading Trend Patterns in all Provinces of China. For this the graph generated above was plotted for all the provinces. The Graph was made in MS-Excel and the results are fetched underneath.

# In[ ]:


#Viewing the generated graphs for the reported cities in China

get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(15,5))
img=mpimg.imread('../input/china-covid19-data/Anhui.png')
imgplot = plt.imshow(img)
plt.show()

plt.figure(figsize=(15,5))
img=mpimg.imread('../input/china-covid19-data/Beijing.png')
imgplot = plt.imshow(img)
plt.show()

for i in range(1,17):
    plt.figure(figsize=(15,5))
    img=mpimg.imread('../input/china-covid19-data/Screenshot ({}).png'.format(303+i))
    imgplot = plt.imshow(img)
    plt.show()


# <H1> Analyzing the Graph Generated </H1>
# 
# For all of the Graphs of Provicnes Generated above, the Graph is divided into 3 sections. 
# 
# 1. The vertical bars represent the total confimed COVID-19 Cases (Tick Labels on LHS of Y axis inside the figure).
# 2. The X axis for all the above graphs denoted the dates on which the cases were reported.
# 3. The (RHS) Y axis denoted the % daily growth rate of newer confirmed COVID-19 Cases.
# 
# The stacked bars, mentioned in chart above shows the confirmed cases of COVID-19 and it's progression with time. The red line chart shows the growth rate of newer cases reported daily.
# 
# The newer cases have been calculated here by the formula :
# 
# Growth Rate = ((Newer Cases / Older cases) - 1) * 100
# 
# 4. The four translucent rectangle boxes in the chart highlights the 4 weeks since the first outbreak of COVID-19 was reported.
# 
# 
# <h1> Trends Observed in the Underlying Graphs for China </h1>
# 
# * For all of the provinces displayed above, the frequency of COVID-19 Cases reduced drastically after Week 4 of disease outbreak. The daily growth rate of cases declined.
# 
# <h1> Hunting for the possible cases for decline in growth rate of new COVID-19 Cases in China post week 4 </h1>
# 
# We search for the text in literature to find the best papers regarding the same. This is done by using prediction_text()

# In[ ]:


#Searching for the relevant articles
prediction_text()


# <h1> Please Note </h1>
# 
# This notebook is still in it's development Phase. We would add more details to the same with the later versions. Feel free to share much queries/information.
# 
# Developers:
# 
# 1. Aman Kumar - [linkedin.com/in/amankumar01/](http://)
# 2. Keertikesh Rajkiran

# In[ ]:




