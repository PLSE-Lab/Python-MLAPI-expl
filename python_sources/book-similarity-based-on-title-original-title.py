# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from nltk import word_tokenize, pos_tag
from multiprocessing import Process, Manager


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
files = check_output(["ls", "../input"]).decode("utf8").split("\n")

#tags -> important tags from nltk library
important_tags = ["CD","FW","JJ","JJR","JJS","NN","NNP","NNPS","NNS","VBD","VB","VBG","VBN","VBP","VBZ"]

books_file = files[1]
books = open("../input/"+books_file)
book_information = []

print("Reading data")
# Header
header = books.readline().replace("\n","").split(",")
books = list(books)

# Constant time of index lookup
value_index = {}
for index in range(len(header)):
    value_index[header[index]] = index

vector_model = []
norm_vector = [0]*len(books)

title_index = value_index["title"]
original_title_index = value_index["original_title"]

print("Generating Vector Model")
# Vector Model Generation
for book in books:
    
    book = book.replace("\n","")
    
    quote_fields = book.split('"')
    # Deal with Kommas inside titles or original titles
    if len(quote_fields) != 0:
        book = ""
        for field in range(len(quote_fields)):
            if field % 2 == 1:
                quote_fields[field] = quote_fields[field].replace(","," ; ")
            book = book + quote_fields[field]
    
    book = book.split(",")
    
    book_information.append(book)
    
    if book[title_index].find('"') != -1:
        book[title_index] = book[title_index].replace('"'," ")
    
    index = book[title_index].find("#")
    if index != -1:
        second_index = book[title_index].find(" ",index)
        book[title_index] = book[title_index][:index]+ book[title_index][second_index:]

    if book[original_title_index].find('"') != -1:
        book[original_title_index] = book[original_title_index].replace('"'," ");
    
    
    # Word tokenization
    words = word_tokenize(book[title_index])
    words = words + word_tokenize(book[original_title_index])
    words = list(set(words))
    #Pos Tagging
    for tagged in pos_tag(words):

    #filtering tags
        if tagged[1] in important_tags:
            da_word = tagged[0].lower()
            # Avoid repeated words
            if da_word not in vector_model:
                vector_model.append(da_word)

vector_model = sorted(vector_model)
similar_books = []


print("Generating book vectors")
index_completed = 0
num_books = len(books)

# Generating Vectors
for book in books:
    vm = []
    book = book.replace("\n","")
    
    quote_fields = book.split('"')
    # Deal with Kommas inside titles or original titles
    if len(quote_fields) != 0:
        book = ""
        for field in range(len(quote_fields)):
            if field % 2 == 1:
                quote_fields[field] = quote_fields[field].replace(","," ; ")
            book = book + quote_fields[field]
    
    book = book.split(",")

    if book[title_index].find('"') != -1:
        book[title_index] = book[title_index].replace('"'," ")
    
    index = book[title_index].find("#")
    if index != -1:
        second_index = book[title_index].find(" ",index)
        book[title_index] = book[title_index][:index]+ book[title_index][second_index:]

    if book[original_title_index].find('"') != -1:
        book[original_title_index] = book[original_title_index].replace('"'," ");
   
    # Word tokenization
    words = word_tokenize(book[title_index])
    words = words + word_tokenize(book[original_title_index])
    words = list(set(words))
    #Pos Tagging
    for tagged in pos_tag(words):
    #filtering tags
        if tagged[1] in important_tags:
            da_word = tagged[0].lower()
            index = vector_model.index(da_word)
            vm.append(index)
    similar_books.append(vm)
    print("{0}% completed".format(index_completed*100.0/num_books))
    index_completed += 1

#Computing Once the Norm of the vectors
for each in range(len(similar_books)):
	norm_vector[each] = len(similar_books[each])


#Manager result
manager = Manager()
similarity = manager.dict()

# Dot product
def dot_product(v1,v2):
	s1 = set(v1)
	s2 = set(v2)
	res = s1 & s2
	return len(res)

print("Computing Similarity")
similarity = {}
for each in range(len(books)):
    sim = []
    for keach in range(len(books)):
        if each != keach: 
            each_vector = similar_books[each]
            keach_vector = similar_books[keach]
            dot = dot_product(each_vector,keach_vector)
            each_norm = norm_vector[int(each)]
            keach_norm = norm_vector[int(keach)]
            if (each_norm*keach_norm) != 0:
            	res = dot/(each_norm*keach_norm)
            	if res != 0:
            		sim.append((int(keach),res))
    similarity[each] = (sorted(sim,key=lambda x:x[1]))[::-1]
    print("----------------- {0}%".format(each*100.0/len(books)))

fh = open('recommendations.txt',"w")

for each in similarity:
    fh.write("Recommendation for book: '"+book_information[each][title_index]+"'\n")
    for index, sim_val in similarity[each]:
        fh.write(" Similarity Value: "+str(sim_val)+" - "+book_information[index][title_index]+"\n")

print("Results can be sound at /output/recommendation.txt")


# Any results you write to the current directory are saved as output.
