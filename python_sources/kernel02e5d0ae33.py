# Movie recomendations 

import pandas as pd
import numpy as np
import io
import os
from scipy import spatial # this is to get the cosines between vectors


##############################################################################
##Functions below:
##############################################################################
def clean_genres(x):
    counter = 0
    for index, row in x.iterrows():
        q = row['genres']
        char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{}[]":,'
        for l in char:
            q = q.replace(l, "").replace("    ", ',').rstrip(" ")
            q = str(q)
            q = q.strip(" ")
        x.iloc[counter, 4] = q
        counter += 1
    return x

def find_generos(x):
    generos = []
    for index, row in x.iterrows():
        gen_list = row['genres'].replace(" ", "").split(',')
        for i in gen_list:
            if i not in generos:
                generos.append(i)
    return generos

def movie_dict(x):
    movies_dict = {}
#    ['28', '12', '14', '878', '80', '18', '53', '16', '10751', '37', '35', '10749', '27', '9648', '36', '10752', '10402', '99', '10769', '10770', '']
    movie_id = 0
    for index, row in x.iterrows():
        gen_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        gen_list = row['genres'].replace(" ", "").split(',')
        title = row['title']
        vote_average = row['vote_average']
        popularity = row['popularity']
        for i in gen_list:
            if '28' in gen_list:
                gen_vector[0] = 1
            if '12' in gen_list:
                gen_vector[1] = 1
            if '14' in gen_list:
                gen_vector[2] = 1
            if '878' in gen_list:
                gen_vector[3] = 1
            if '80' in gen_list:
                gen_vector[4] = 1
            if '18' in gen_list:
                gen_vector[5] = 1
            if '53' in gen_list:
                gen_vector[6] = 1
            if '16' in gen_list:
                gen_vector[7] = 1
            if '10751' in gen_list:
                gen_vector[8] = 1
            if '37' in gen_list:
                gen_vector[9] = 1
            if '35' in gen_list:
                gen_vector[10] = 1
            if '10749' in gen_list:
                gen_vector[11] = 1
            if '27' in gen_list:
                gen_vector[12] = 1
            if '9648' in gen_list:
                gen_vector[13] = 1
            if '36' in gen_list:
                gen_vector[14] = 1
            if '10752' in gen_list:
                gen_vector[15] = 1
            if '10402' in gen_list:
                gen_vector[16] = 1
            if '99' in gen_list:
                gen_vector[17] = 1
            if '10769' in gen_list:
                gen_vector[18] = 1
            if '10770' in gen_list:
                gen_vector[19] = 1
        if np.linalg.norm(gen_vector) != 0:
            movie_id += 1
            movies_dict[movie_id] = (title, vote_average, popularity, gen_vector, movie_id)
    return movies_dict

def get_distance(movie1, movie2):
    gen_movie1 = movie1[3]
    gen_movie2 = movie2[3]
    movie1_rating = movie1[1] / 10
    movie2_rating = movie2[1] / 10
    movie1_popular = movie1[2] / 875.581305
    movie2_popular = movie2[2] / 875.581305
    distance_popular = abs(movie1_popular - movie2_popular)
    distance_rating = abs(movie1_rating - movie2_rating)
    distance_gen = spatial.distance.cosine(gen_movie1, gen_movie2)
    distance = (0.5 * distance_gen) + (0.3 * distance_rating) + (0.2 * distance_popular)
    return distance

def rating_info(movie_dict): # this just make a list with the ratings of the movies. I want to know what is the average, the min and the max
    ratings = []
    for i in movie_dict:
        #print(movie_dict[i][1])
        ratings.append(movie_dict[i][1])
    print("the average is: ", np.mean(ratings))
    print("the minumum is: ", np.min(ratings))
    print("the max is: ", np.max(ratings))
    return ratings

def popularity_info(movie_dict):
    popularity = []
    for i in movie_dict:
        popularity.append(movie_dict[i][2])
    print("the average is: ", np.mean(popularity))
    print("the minumum is: ", np.min(popularity))
    print("the max is: ", np.max(popularity))
    return popularity
    
def neighbors(movie_dict, N): #takes the dicctionary of movies and the id number of the movie that we are interested in
    neighbors = pd.DataFrame(columns=['movie','distace'])
    counter = 0
    for i in movie_dict:
        counter += 1
        title = movie_dict[i][0]
        distance = get_distance(movie_dict[N], movie_dict[i])
        #print(distance)
        neighbors.loc[counter] = [title, distance]
    neighbors = neighbors.sort_values(by=['distace'])
    return neighbors.head(6)

def movie_list_show(movie_dict):
    for i in movie_dict:
        print(movie_dict[i][0], "---->", movie_dict[i][4])
    return 0

def movie_find(movie_dict, name):
    for i in movie_dict:
        title = movie_dict[i][0]
        id = movie_dict[i][4]
        if name in title:
            print(title, "---->", id)
        
                 
##############################################################################
###Code starts here
##############################################################################
path_to_file = "../input/tmdb-5000-movies/tmdb_5000_movies.csv" #Path to the tmdb_5000_movies.csv file
full_table = pd.read_csv(path_to_file, header = 0) #Reads the Full Table (I am not going to use the full table)
table = pd.DataFrame(full_table[['title', 'vote_average', 'vote_count', 'popularity', 'genres']]) #Gets the Columns we are interested in

clean_table = pd.DataFrame(clean_genres(table)) # Cleans the table
#print(find_generos(clean_table)) # This is just to check how many and which generos we have
movie_dict = movie_dict(clean_table) # makes a movie dictionary with the info: (title, vote_average, popularity, gen_vector, movie_id) 

##############################################################################
###USER PART OF THE CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##############################################################################
#movie_list_show(movie_dict)  #prints to the screen the list of movies and the id (Do not use the id in the csv file to look for movies. It wont work)
#movie_find(movie_dict, "Cars") # Finds movie with similar name to the one you have in mind

x = neighbors(movie_dict, 567) # by spesifying the id of the movie, the code gives you back a top 5 of similar movies.
print(x)