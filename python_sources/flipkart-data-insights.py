import pandas as pd
import numpy as np
import re

#load files
pre_df=pd.read_csv(r'..\flipkart_com-ecommerce_sample.csv')

#split the data on >> and strip special characters
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('[]'))
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('"'))
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.split('>>'))

#delete unwanted columns
del_list=['crawl_timestamp','product_url','image','description','product_specifications']
pre_df=pre_df.drop(del_list,axis=1)

#create product categories  
pre_df['product_category']=pre_df['product_category_tree'].map(lambda x:x[0])   
pre_df['product_category_level1']=pre_df['product_category_tree'].map(lambda x:x[1] if x[1:2] else "")

#calculate discount percentage
pre_df['disct_percentage']=(pre_df['retail_price']-pre_df['discounted_price'])*100/pre_df['retail_price']
pre_df['disct_percentage']=np.round(pre_df['disct_percentage'],2)

#replace empty rating values with np.nan
pre_df['product_rating']=pre_df['product_rating'].replace('No rating available',np.NaN)
pre_df['overall_rating']=pre_df['overall_rating'].replace('No rating available',np.NaN)

#selecting only rated rows
rating_df=pre_df[pre_df['overall_rating'] != np.NaN]


rating_df.product_rating=pd.to_numeric(rating_df.product_rating)
#rating_df['aggr']=rating_df.product_rating.mean()


#grouping based on product name and category and subcategory
#top_ten_most_rated_product=rating_df.groupby(['product_category','product_category_level1'],as_index=False).agg({'product_rating':['count',np.mean],'overall_rating':'count'})
top_ten_most_rated_product=rating_df.groupby(['product_category','product_category_level1'],as_index=False).agg({'product_rating':np.mean,'overall_rating':'count'})
top_ten_most_rated_product=top_ten_most_rated_product.sort_values(['overall_rating'],ascending=[False])
top_ten_most_rated_product=top_ten_most_rated_product.rename(columns={'product_rating':'AVG Rating'})
#top_ten_most_rated_product=top_ten_most_rated_product.head(10)

#Selecting only NaN value rows
Non_rating_df=pre_df[pd.isnull(pre_df['overall_rating'])]
Non_rating_df['NON_Rated_No_Items']=1

#grouping based on product name and category and subcategory
top_ten_non_rated_product=Non_rating_df.groupby(['product_category','product_category_level1'],as_index=False).agg({'NON_Rated_No_Items':'count'})
top_ten_non_rated_product=top_ten_non_rated_product.sort_values(['NON_Rated_No_Items'],ascending=[False])
#top_ten_non_rated_product=top_ten_non_rated_product.head(10)

#Merging both Non Rated and Rated products to find the product which is most likely to be rated
Likely_rated_product=top_ten_most_rated_product.merge(top_ten_non_rated_product,on=['product_category','product_category_level1'], how='inner')
Likely_rated_product['Total_No_Items']=Likely_rated_product['overall_rating'] + Likely_rated_product['NON_Rated_No_Items']
Likely_rated_product['most_likely_percentage']=(Likely_rated_product['Total_No_Items']-Likely_rated_product['NON_Rated_No_Items'])*100/Likely_rated_product['Total_No_Items']
Likely_rated_product=np.round(Likely_rated_product,2)

top_discounted_Brands=rating_df.groupby(['brand','product_category_level1'],as_index=False).agg({'disct_percentage':np.mean})
top_discounted_Brands=np.round(top_discounted_Brands,2)
top_discounted_Brands=top_discounted_Brands.reindex()

top_discounted_Brands=top_discounted_Brands.sort_values(by='disct_percentage',ascending=False)

pre_df.to_csv(r'..\pre.csv')
rating_df.to_csv(r'..\rating_df.csv')

top_ten_most_rated_product.to_csv(r'..\top_ten_most_rated_product.csv')
top_ten_non_rated_product.to_csv(r'..\top_ten_non_rated_product.csv')
Likely_rated_product.to_csv(r'..\Likely_rated_product.csv')
top_discounted_Brands.to_csv(r'..\top_discounted_Brands.csv')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
top_ten_non_rated_product.head(15).plot.bar(title="TOP NON-RATED PRODUCTS",x='product_category_level1')

top_ten_most_rated_product.head(15).plot.bar(title="TOP RATED PRODUCTS",x='product_category_level1')

Likely_rated_product=Likely_rated_product.head(10)
Likely_rated_product[['Total_No_Items','NON_Rated_No_Items','overall_rating','AVG Rating','most_likely_percentage','product_category']].plot.bar(title="TOP LIKELY RATED PRODUCTS",x=Likely_rated_product['product_category_level1'])

top_discounted_Brands.head(15).plot.bar(title="TOP DISCOUNTED BRANDS",x='brand')

top_discounted_Brands.columns
