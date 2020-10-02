#!/usr/bin/env python
# coding: utf-8

# # 02 Data Structures

# - Tuples
# - Lists
# - Sets
# - Dictionaries

# # Tuples

# > immutable orderd sequences

# In[ ]:


genres = ("pop", "rock", "soul", "hard rock", "soft rock",                 "R&B", "progressive rock", "disco") 
print(f'Genres: {genres}')
print(f'Type of genres: {type(genres)}')
print(f'Length of genres: {len(genres)}')


# ## Indexing

# In[ ]:


genres[0]


# In[ ]:


genres[-1]


# ## Concatenation

# In[ ]:


disco = ("disco",10,1.2)
rock = ("rock",10)

disco_rock = disco + rock

print(f'Disco: {disco}')
print(f'Rock: {rock}')
print(f'Disco Rock: {disco_rock}')


# ## Slicing

# In[ ]:


genres[3:6]


# In[ ]:


genres[0:2]


# In[ ]:


genres.index('disco')


# ## Sorting

# In[ ]:


ratings = (0, 9, 6, 5, 10, 8, 9, 6, 2)
ratings


# In[ ]:


ratings_sorted = tuple(sorted(ratings))
ratings_sorted


# In[ ]:


ratings_reversed = tuple(sorted(ratings,reverse=True))
ratings_reversed


# ## Tuples: Immutable

# In[ ]:


# ratings[2] = 4
# TypeError: 'tuple' object does not support item assignment


# In[ ]:


nested = (1,2,("pop","rock"),(3,4),("disco",(1,2)))
nested


# In[ ]:


nested[2]


# In[ ]:


nested[2][1]


# In[ ]:


nested[2][1][0]


# # Lists

# > mutable orderd sequences

# In[ ]:


a_list = ["John Smith",10.1,1982]
print(f'a_list: {a_list}')
print(f'Type of a_list: {type(a_list)}')
print(f'Length of a_list: {len(a_list)}')


# In[ ]:


a_list[1]


# In[ ]:


a_list = ["John Smith",10.1,1982]
b_list = ["Jane Smith",9.1,1988]

ab_list = a_list + b_list

print(f'a_list: {a_list}')
print(f'b_list: {b_list}')
print(f'ab_list : {ab_list}')


# In[ ]:


a_list_added = a_list + ["pop",10]
a_list_added


# In[ ]:


a_list = ["John Smith",10.1,1982]
a_list


# In[ ]:


a_list.extend(["pop",10])


# In[ ]:


a_list


# In[ ]:


a_list = ["John Smith",10.1,1982]
a_list


# In[ ]:


a_list.append(["pop",10])


# In[ ]:


a_list


# In[ ]:


A = ["disco",10,1.2]
A


# In[ ]:


A[0] = "hard rock"
A


# In[ ]:


del(A[0])
A


# ## Convert String to List

# In[ ]:


"hard rock".split()


# In[ ]:


"A,B,C,D".split(",")


# ## Aliasing

# In[ ]:


A = ["hard rock",10,1.2]
A


# In[ ]:


B = A
B


# In[ ]:


A[0]


# In[ ]:


B[0]


# In[ ]:


A[0] = "banana"
A[0]


# In[ ]:


B[0]


# ## Clone

# In[ ]:


A = ["hard rock",10,1.2]
A


# In[ ]:


B = A[:]
B


# In[ ]:


A[0]


# In[ ]:


B[0]


# In[ ]:


A[0] = "banana"
A[0]


# In[ ]:


B[0]


# # Sets

# > unorded collection of unique elements

# In[ ]:


album_list = ['Michael Jackson',"Thriller","Thriller",1982]

print(f'Album list: {album_list}')
print(f'Type of Album list: {type(album_list)}')
print(f'Length of Album list: {len(album_list)}')


# In[ ]:


album_set = set(album_list)
     
print(f'Album set: {album_set}')
print(f'Type of Album set: {type(album_set)}')
print(f'Length of Album set: {len(album_set)}')


# ## Set Operations

# In[ ]:


A = {"Thriller","Back in Black","AC/DC"}
A


# In[ ]:


A.add("NSYNC")
A


# In[ ]:


A.remove("NSYNC")
A


# In[ ]:


"AC/DC" in A


# In[ ]:


"Who" in A


# ### Set Operations

# In[ ]:


album_set_1 = {'AC/DC', 'Back in Black', 'Thriller'}
album_set_1


# In[ ]:


album_set_2= {'AC/DC', 'Back in Black', 'The Dark Side of the Moon'}
album_set_2


# In[ ]:


intersection = album_set_1 & album_set_2
intersection


# In[ ]:


album_set_4 = album_set_1.union(album_set_2)
album_set_4


# In[ ]:


intersection.issubset(album_set_1)


# In[ ]:


album_set_4.issubset(album_set_1)


# In[ ]:


album_set_4.issuperset(album_set_1)


# In[ ]:


album_set_1.difference(album_set_2)  


# In[ ]:


album_set_2.difference(album_set_1)  


# In[ ]:


album_set_1.intersection(album_set_2)  


# # Dictionaries

# > collection of elements that consist of keys and values

# In[ ]:


release_year = {"Thriller": "1982", 
                "Back in Black": "1980", 
                "The Dark Side of the Moon": "1973", 
                "The Bodyguard": "1992", 
                "Bat Out of Hell": "1977", 
                "Their Greatest Hits (1971-1975)": "1976", 
                "Saturday Night Fever": "1977", 
                "Rumours": "1977"}

print(f'Release year: {release_year}')
print(f'Type of ARelease year: {type(release_year)}')
print(f'Length of Release year: {len(release_year)}')


# In[ ]:


release_year['Thriller']


# In[ ]:


print(f'Release year keys: {release_year.keys()}')
print(f'Release year values: {release_year.values()}')


# In[ ]:


release_year['Graduation'] = '2007'
release_year


# In[ ]:


# Delete entries by key
del(release_year['Thriller'])
del(release_year['Graduation'])
release_year


# In[ ]:


'The Bodyguard' in release_year


# In[ ]:


'Starboy' in release_year

