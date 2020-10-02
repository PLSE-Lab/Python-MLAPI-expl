'''
Dated: may03-2019
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for product recommendation using collaborative filtering
'''

import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collab_filtering(product_list):
	ratings=pd.read_csv('../input/new-ratings.csv')
	# print(new-ratings.head(15))

	products=pd.read_csv('../input/new-products.csv')
	# print(new-products.head(15))

	product_ratings = pd.merge(products, ratings)
	# print('product_ratings.info():\n', product_ratings.info())

	ratings_matrix = ratings.pivot_table(index=['productId'],columns=['userId'],values='rating').reset_index(drop=True)
	ratings_matrix.fillna( 0, inplace = True )
	# print(ratings_matrix.head(15))

	product_similarity=cosine_similarity(ratings_matrix)
	np.fill_diagonal( product_similarity, 0 ) 
	product_similarity

	ratings_matrix = pd.DataFrame( product_similarity )
	# print(ratings_matrix.head(15))

	i=0
	for user_inp in product_list:
		i = i+1
		try:
			#user_inp=input('Enter the reference product name based on which recommendations are to be made: ')
			# user_inp="Executive Decision (1996)"
			inp=products[products['name']==user_inp].index.tolist()
			inp=inp[0]

			products['similarity'] = ratings_matrix.iloc[inp]
			# print(products.head(5))

		except:
			print("Sorry, the product is not in the database!")

		print("Recommended products based on your choice of ",user_inp ,": \n", products.sort_values( ["similarity"], ascending = False )[1:10])
		# outfname = "output-"%i+".csv"
		outfname = "output.csv-{0}".format(i)
		products.sort_values( ["similarity"], ascending = False )[1:10].to_csv(outfname)
    
if __name__ == "__main__":
	product_list = ["Executive Decision (1996)", "Cosi (1996)", "Office Space (1999)", "Hamlet (2000)", "Butterfield 8 (1960)"]
	collab_filtering(product_list)