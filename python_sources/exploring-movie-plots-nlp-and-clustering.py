#!/usr/bin/env python
# coding: utf-8

# # Recommend Movies With Similar Plots
# ### Matching the plot descriptions with NLP
# 
# I've written a movie recommendation notebook already which uses collaborative filtering, finding movies you might like based on your previous likes and the likes of others. But that can suffer from what's known as a cold start - you need have a list of films you have already voted on, as well as the votes of others!
# 
# This model is just another means about how we might go about finding something of interest. There's many issues with doing it this way as we're just matching the plot description. It could find a terrible with just a similar plot (think Rocky vs Rocky V). It's always fun to play around with though.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode()
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


data = pd.read_csv("/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv", usecols=['Release Year','Title','Plot'])
data.drop_duplicates(inplace=True)
data['Title'] = data['Title'].astype(str)
data['Title'] = data['Title'].apply(lambda x: x.strip())
data.head()


# It'll certainly be useful to see what ratings these films have so lets find an IMDB dataset on here that has ratings and left join it based on the title and date of release.

# In[ ]:


imdb_titles = pd.read_csv("../input/imdb-extensive-dataset/IMDb movies.csv", usecols=['title','year','genre'])
imdb_ratings = pd.read_csv("../input/imdb-extensive-dataset/IMDb ratings.csv", usecols=['weighted_average_vote'])

ratings = pd.DataFrame({'Title':imdb_titles.title,
                    'Release Year':imdb_titles.year,
                    'Rating': imdb_ratings.weighted_average_vote,
                    'Genre':imdb_titles.genre})
ratings.drop_duplicates(subset=['Title','Release Year','Rating'], inplace=True)
ratings.head()


# In[ ]:


data = data.merge(ratings, how="left", on=['Title','Release Year'])


# In[ ]:


data.drop_duplicates(inplace=True)


# In[ ]:


print(f'Movies with missing data: {data.Rating.isna().mean()*100:.1f}%')


# Looking quickly at the number of rows that didn't match with the IMDB dataset shows just under half that won't have a rating. Film titles can vary greatly depeding on region and if translated to english or not. Plus there is the punctuation that might not match. The year could also be off my a margin of one or two years depending on what region it's basing it's release.

# In[ ]:


#with nlp.disable_pipes():
#    vectors = np.array([nlp(data.Plot).vector for idx, data in data.iterrows()])
    
#vectors.shape


# We'll save these vectors as they take an awfully long time

# In[ ]:


import pickle
model_dir = "/kaggle/input/vector/"
model_file = "vectors.sav"
#with open(model_file,mode='wb') as model_f:
#    pickle.dump(vectors,model_f)


# In[ ]:


with open(model_dir + model_file,mode='rb') as model_f:
    vectors = pickle.load(model_f)


# Lets find a film we want to try and match. I love Dunkirk so lets see if that's in our dataframe.

# In[ ]:


data[data.Title.str.contains('Dunkirk')]


# Excellent. Now to assign the wiki desciption to the variable plot and take a peak at it's description.

# In[ ]:


plot = data.Plot[21598]
plot[:500]


# In[ ]:


def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

X = vectors
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X)


# The cosine similarity will measure the similarities between the text vector and the vector of the mean, giving a value between -1 and 1, with 1 being an exact match.
# 
# I'm also going to use principal component analysis on the vectors because why not, and I thought it might be fun to graph these results in 3D space. Do be aware that the PCA vectors will not allign exactly with their cosine similarity. What I mean is a movie with the highest cosine to another will not neccessarily be the closest to that movie when plotting their PCA values.

# In[ ]:


def movie_similarities(plot, **kwargs):
        
    # optional parameters
    params = {'year' : 1900, # list movies after a certain date
              'new_plot' : True, # if the plot isn't in the dataset
              'n' : 10, # no. results
              'rating' : 0.0, # list movies rated above a certain IMDB score
              'nan' : True} # don't include titles without an IMDB score
    
    for key, value in kwargs.items():
        params[key] = value
    
    x = int(params['new_plot'])
    
    movie_vec = nlp(plot).vector
    vec_mean = vectors.mean(axis=0)
    centered = vectors - vec_mean
    sims = np.array([cosine_similarity(movie_vec - vec_mean, vec) for vec in centered])
    
    movie_index = []
    movie_title = []
    movie_year = []
    movie_rating = []
    movie_cosine = []
    movie_genre = []
    
    pca_0 = []
    pca_1 = []
    pca_2 = []
    
    for i in sorted(sims)[-2+x::-1]:
        if not params['nan'] and np.isnan(data.iloc[np.where(sims == i)[0][0]]["Rating"]):
            pass
        
        elif data.iloc[np.where(sims == i)[0][0]]["Release Year"] >= params['year']                 and (data.iloc[np.where(sims == i)[0][0]]["Rating"] >= params['rating']                 or np.isnan(data.iloc[np.where(sims == i)[0][0]]["Rating"])):
            
            index = np.where(sims == i)[0][0]
            
            movie_index.append(index)
            movie_title.append(data.iloc[index]['Title'])
            movie_year.append(data.iloc[index]["Release Year"])
            movie_rating.append(data.iloc[index]["Rating"])
            movie_genre.append(data.iloc[index]["Genre"])
            movie_cosine.append(round(i,2))
            
            pca_0.append(pca_result[:,0][index])
            pca_1.append(pca_result[:,1][index])
            pca_2.append(pca_result[:,2][index])
                        
            params['n'] -= 1
            
        if params['n'] == 0:
            break
            
    return  pd.DataFrame({'Title':movie_title,
                        'Year':movie_year,
                        'IMDB':movie_rating,
                        'Cosine':movie_cosine,
                        'Genre':movie_genre,
                        'pca_0':pca_0,
                        'pca_1':pca_1,
                        'pca_2':pca_2},
                        index=movie_index)


# The above function lists n number of films with the closest cosine similarity. I've since added optional parameters to list films released from a certain date, and to add a plot not in the original dataset.

# In[ ]:


movie_similarities(plot, year=1990).iloc[1:,:5]


# Here we have the top ten films that are most similar to the plot of Dunkirk.
# I've not seen U-571, and it's got a fresh rating from Rotten Tomatoes. Looks just up my street, fantastic.
# 
# I suppose Speed 2: Cruise Control has boats in it... We can see how this method really doesn't account for genres which is a large feature when trying to find a similiar movie we might like.

# In[ ]:


data[data.Title.str.contains('Gladiator')]


# Another film I like is Gladiator (2000) so lets find some similar films again.

# In[ ]:


movie_similarities(data.Plot[13665]).iloc[1:,:5]


# I can certainly see the similarity with Gladiator and The Legend of Hercules but sadly the latter looks like trash with a 4.2% rating on IMDB.
# 
# 

# Finally, lets have some fun and write a movie plot of our own and see what film might me most similar to it.

# In[ ]:


plot = """A young man with no options left must seek a path to glory by uncovering his great grandfather's 
chest. Revealing a time machine that transports him back to the world war 2. Here hitler and his army must 
use exotic snakes to hypnotise dragons into obeying their command. The confusion of such events unravel to 
reveal our hero has travelled into another dimension where his dream girl awaits for him."""


# In[ ]:


movie_similarities(plot).iloc[:,:5]


# Taking a peak at the plot for The Mad Monk shows that maybe the fact they both mention dragons is why is was listed as most similar.

# In[ ]:


movie_similarities(data.Plot[20779], n=40).iloc[:,:]


# Now what I'd like to do is plot the list of movies above (based on their similarity to the first Harry Potter film) based on their PCA values. We would hope, that they are grouped closer together then the other films listed.

# In[ ]:


def plot_pca(plot, **kwargs):
    df = movie_similarities(plot, **kwargs)
    text = df.Title + '<br>IMDB: ' + df.IMDB.astype(str) + '<br>Cosine: ' + round(df.Cosine,2).astype(str)
    fig = go.Figure()

 

    fig.add_trace(go.Scatter3d(
        x=df.pca_0[1:11],
        y=df.pca_1[1:11],
        z=df.pca_2[1:11],
        name='Harry Potter Films',
        hovertemplate='%{text}<br><extra></extra>',
        text = text[1:11],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=df.pca_0[:1],
        y=df.pca_1[:1],
        z=df.pca_2[:1],
        name=df.Title.values[0],
        hovertemplate='%{text}<br><extra></extra>',
        text = text[:1],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8
        )
    ))   
        
    fig.add_trace(go.Scatter3d(
        x=df.pca_0[11:],
        y=df.pca_1[11:],
        z=df.pca_2[11:],
        name='Other Films',
        hovertemplate='%{text}<br><extra></extra>',
        text = text[11:],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8,
        )
    ))
        
    fig.update_layout(title=f'Movies similar to {df.Title.values[0]}')
    
    fig.show()
    
plot_pca(data.Plot[20779], n=40)


# We can see the Harry Potter films (in blue) are bunched into one corner. What's interesting though is the film Gambit looks to be the closest to Harry Potter and the Sorcerer's Stone despite it being one of the last listed films on the 40 movies with the cosine similarity closest to 1.

# In[ ]:


pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(vectors)
X_reduced.shape

# run kmeans with many different k
distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    if k % 5 == 0:
        print('Found distortion for {} clusters'.format(k))


# In[ ]:





X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[ ]:


k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_reduced)
data['y'] = y_pred


# In[ ]:


df = movie_similarities(data.Plot[20779], n=100)
x = df.merge(data['y'].to_frame(), left_index=True, right_index=True)
x.sort_values(by='Cosine', ascending=False).head(10)


# In[ ]:


df = data
df['pca0'] = pca_result[:,0]
df['pca1'] = pca_result[:,1]
df.dropna(inplace=True)
df['y'] = df['y'].astype(str)
text = df.Title + '<br>IMDB: ' + df.Rating.astype(str) + '<br>Genre: ' + df.Genre
 
fig = px.scatter(df, x="pca0", y="pca1", color="y", hover_data=['Title','Genre'])
fig.show()


# In[ ]:


df.Genre


# In[ ]:


genres_df = df['Genre'].apply(lambda x: x.split(",")).apply(pd.Series)


# In[ ]:


a = pd.DataFrame({'y':df.y, 'Genre':genres_df[0]}).reset_index()
b = pd.DataFrame({'y':df.y, 'Genre':genres_df[1]}).reset_index()
c = pd.DataFrame({'y':df.y, 'Genre':genres_df[2]}).reset_index()

genres_df = a.append(b).append(c).dropna()


# In[ ]:


genres_df


# In[ ]:


genres_df.Genre = genres_df.Genre.apply(lambda x: x.strip())
x = genres_df.groupby(['y','Genre']).count().reset_index()
x


# In[ ]:


from plotly.subplots import make_subplots
di = {"type": "Scatterpolar"}
fig = make_subplots(rows=5, cols=2,
                    subplot_titles=(["Type "+str(i) for i in range(10)]), 
                    specs=np.array([di for i in range(10)]).reshape(5,2).tolist())
                                          
y=0
for i in range(5):
    for j in range(2):
        fig.add_trace(go.Scatterpolar(
              r=x[(x.y == str(y)) & (x.Genre != 'Drama')]['index'],
              theta=x[(x.y == str(y)) & (x.Genre != 'Drama')].Genre.unique(),
              fill='toself',
              name=str(y)
        ), row=i+1, col=j+1)
        y += 1

fig.update_traces()
fig.update_layout(height=1600)
fig.show()


# In[ ]:


x.sort_values(by=['y','index'], ascending=False).groupby('y')['Genre'].head(10).value_counts().plot(kind='bar')


# For the ten highest listed genres for each class we can see Crime, Drama, Thriller, and Comedy appear in every class.

# In[ ]:


x.groupby('Genre').sum().sort_values(by='index', ascending=False).plot(kind='bar')


# From the most listed Genre's we can see these are also the ones that appear in each class.

# In[ ]:


not_in = x.sort_values(by=['y','index'], ascending=False).groupby('y')['Genre'].head(10).value_counts().index[:8]


# I'm going to take another look at the radar plots but removing the eight most common genre's to get a better insight into the shape of the ach class.

# In[ ]:


from plotly.subplots import make_subplots
di = {"type": "Scatterpolar"}
fig = make_subplots(rows=5, cols=2,
                    subplot_titles=(["Type "+str(i) for i in range(10)]), 
                    specs=np.array([di for i in range(10)]).reshape(5,2).tolist())
                                          
y=0
for i in range(5):
    for j in range(2):
        fig.add_trace(go.Scatterpolar(
              r=x[(x.y == str(y)) & (~x.Genre.isin(not_in))]['index'],
              theta=x[(x.y == str(y)) & (~x.Genre.isin(not_in))].Genre.unique(),
              fill='toself',
              name=str(y)
        ), row=i+1, col=j+1)
        y += 1

fig.update_traces()
fig.update_layout(height=1600)
fig.show()

