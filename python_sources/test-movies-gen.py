# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datetime
import csv
import urllib.request

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

with open("../input/movies_metadata.csv") as movies_file:
    movies_frame = pd.read_csv(movies_file)
    
movie_ids = set()
with open("./movies.csv", 'w') as output_movies:
    fieldnames = ['id', 'name', 'genre', 'release_date']
    movie_writer = csv.DictWriter(output_movies, fieldnames)
    # movie_writer.writeheader()
    for index, frame in movies_frame.iterrows():
        try:
            if frame['original_language'] != 'en':
                continue
            movie_id = int(frame['id'])
            movie_title = frame['original_title']
            movie_genre = random.choice(eval(frame['genres']))['name']
            movie_release_date = frame['release_date']#datetime.datetime.strptime(frame['release_date'], '%Y-%m-%d').date()
            
            assert movie_id and movie_title and movie_genre and movie_release_date
        except Exception as e:
            continue
        
        movie_writer.writerow({'id': movie_id, 'name': movie_title, 'genre': movie_genre, 'release_date':movie_release_date})
        movie_ids.add(movie_id)

movie_ids = tuple(movie_ids)

with open("../input/credits.csv") as credits_file:
    credits_frame = pd.read_csv(credits_file)
    
def rand_birth_date():
    year = random.randint(1950, 2000)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime.date(year, month, day)
    
actors = {}
leads = []
for index, frame in credits_frame.iterrows():
    try:
        movie_id = int(frame['id'])
        if movie_id not in movie_ids:
            continue
        cast = eval(frame['cast'])
        for member in cast:
            name = member['name']
            gender = 'm' if int(member['gender']) == 2 else 'f'
            cast_id = int(member['id'])
            year = random.randint(1950, 2000)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            birth_date = rand_birth_date()
            actors[cast_id] = {'id':cast_id, 'name':name, 'gender':gender, 'date_of_birth':birth_date}
            leads.append({'actor_id':cast_id, 'movie_id':movie_id})
    except Exception as e:
        continue
    
print(len(actors))
print(len(leads))
    
with open("./actors.csv", 'w') as output_actors:
    fieldnames = ['id', 'name', 'gender', 'date_of_birth']
    actor_writer = csv.DictWriter(output_actors, fieldnames)
    # actor_writer.writeheader()
    
    for cast_id, actor in actors.items():
        actor_writer.writerow(actor)
    
with open("./leads.csv", 'w') as output_leads:
    fieldnames = ['actor_id', 'movie_id']
    lead_writer = csv.DictWriter(output_leads, fieldnames)
    # lead_writer.writeheader()
    
    for lead in leads:
        lead_writer.writerow(lead)


users = []
names = []
for cast_id, actor in actors.items():
    names += actor['name'].split(' ')

def rand_name():
    return ' '.join([random.choice(names), random.choice(names)])
   
print(rand_name())

for x in range(1, 270896):
    users.append({'id': x, 'name': rand_name(), 'date_of_birth': rand_birth_date()})

with open("./users.csv", 'w') as output_users:
    fieldnames = ['id', 'name', 'date_of_birth']
    users_writer = csv.DictWriter(output_users, fieldnames)
    # users_writer.writeheader()
    
    for user in users:
        users_writer.writerow(user)

comment = """Lorem ipsum dolor sit amet consectetur adipiscing elit. Nulla at enim a lorem auctor interdum. Nullam sed dui dolor. Suspendisse tincidunt urna in enim vestibulum vehicula."""

with open("./reviews.csv", 'w') as output_reviews:
    fieldnames = ['user_id', 'movie_id', 'rating', 'comment']
    reviews_writer = csv.DictWriter(output_reviews, fieldnames)
    # reviews_writer.writeheader()
    
    for x in range(0, 500000):
        movie_id = random.choice(movie_ids)
        user_id = random.choice(range(1, 270896))
        rating = random.choice(range(1,10))
        reviews_writer.writerow({'user_id':user_id, 'movie_id':movie_id, 'rating':rating, 'comment':comment})
    
    
    
    
    
    






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    