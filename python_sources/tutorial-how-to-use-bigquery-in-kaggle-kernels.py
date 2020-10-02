#!/usr/bin/env python
# coding: utf-8

# # How to use BigQuery on Kaggle
# Welcome to this step-by-step tutorial showing you how to make the most out of the BigQuery (BQ) integration with Kaggle Kernels! By this point, your Google account should already have Google Cloud Platform (GCP) services enabled and a dataset stored on the GCP console that you want to analyze in your kernel. If you have not done this yet, please visit Kaggle's [documentation](https://www.kaggle.com/docs/notebooks#connecting-kaggle-notebooks-to-google-cloud-services) to get help setting it up. 
# 
# ## About the Example Dataset
# For the purpose of this tutorial, I will be using the publicly available [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) dataset uploaded to my GCP account. Please see the link for documentation on the column variables. The Ames Housing dataset contains 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa as well as their final sales price. You can download the Ames Housing dataset, attach it to your own GCP account and play around with it yourself. Or, feel free to upload any other dataset you want to your GCP account too. It will be private and safely stored. 

# 
# ## Step 1: Link your GCP account to the Kernel
# Head to the right-hand sidebar, click on Settings > BigQuery > Enable > Link an account and sign in to the Google account which you've enabled GCP services for. Once that's linked, that means the Kernel will have access to your BQ account. You can now insert commands to use the [BigQuery API Client library](https://googleapis.github.io/google-cloud-python/latest/bigquery/generated/google.cloud.bigquery.client.Client.html) and access your dataset.

# ## Step 2: Set up BQ Client library and Fetch Your BQ Dataset
# First, find the Project ID that your dataset is uploaded to on BigQuery. You can visit your [GCP console](https://console.cloud.google.com) and view all your projects in the top menu bar. Project IDs simply refer to the name of the project. In my example, I have a project named `my-example-housing-dataset`. That is the Project ID and this project contains the Ames Housing dataset I want to use in this kernel.
# 
# **Optional: Something you may want to do here is set up BQ Magics.**

# In[ ]:


# Identify the project ID containing the desired dataset for analysis in this kernel
PROJECT_ID = 'my-example-dataset'

# Import the BQ API Client library
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location='US')

# Construct a reference to the Ames Housing dataset that is within the project
dataset_ref = client.dataset('ameshousing', project=PROJECT_ID)

# Make an API request to fetch the dataset
dataset = client.get_dataset(dataset_ref)


# ## Step 3: Preview your Dataset
# Now you have a variable `dataset` which contains the BigQuery dataset you want to use in this kernel! For this example, that means I now have my Ames housing dataset. Time to preview the content and make sure everything looks right.

# In[ ]:


# Make a list of all the tables in the dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset
for table in tables:  
    print(table.table_id)


# In[ ]:


# Preview the first five lines of the table
client.list_rows(table, max_results=5).to_dataframe()


# ## Step 4: Write and Run a SQL Query (or many queries)
# Now that we have our data it is time for the main show--writing and executing a SQL query! Perhaps you want to write a query to retrieve a specific subset of the dataset for further use in the kernel. Or maybe you have a specific question about the data and you want it answered via a query. That is all possible within the kernel. Here, I'm going with the latter and composing a query that answers specific questions about the Ames Housing dataset. 

# In[ ]:


# What are the most and least common residential home styles in this dataset? 
# For each home style, how many do and do not have central Air Conditioning?

# Write the query
query1 = """
          SELECT 
            DISTINCT HouseStyle AS HousingStyle, 
            CentralAir AS HasAirConditioning,
            COUNT(HouseStyle) AS Count
          FROM 
            `my-example-dataset.ameshousing.train` 
          GROUP BY 
            HousingStyle, 
            HasAirConditioning
          ORDER BY 
            HousingStyle, 
            HasAirConditioning DESC
        """


# In[ ]:


# Set up the query
query_job1 = client.query(query1)

# Make an API request  to run the query and return a pandas DataFrame
housestyleAC = query_job1.to_dataframe()

# See the resulting table made from the query
print(housestyleAC)


# It looks like one-story homes followed by two-story homes are the most common housing styles in this dataset. That makes sense since Ames, Iowa has a relatively low population density, meaning more land for everyone to have a nice home. The less common housing styles are the unfinished ones, with `Unf` somewhere in the `HousingStyle` name such as 1.5Unf or 2.5 Unf. 
# 
# It is also true that an overwhelming majority of homes, regardless of housing style, have central Air Conditioning (AC). Did you know that Iowa is infamous for its hot and sticky summers? Good thing most of these homes have AC!

# ### What Comes After Querying?
# Note that executing queries returns a nice pandas DataFrame with your query output. With you, you can unleash the rest of your data science skills and do basically anything you'd like with it in Python (or R using the R equivalent BQ Client library)! For example, you could quickly create a graph visualizing the query output using Matplotlib, or create and train a model with XGBoost (though probably not for this exact query I just executed). The sky is the limit. **For example, though not covered in this tutorial, you could add a quick matplotlib bar chart here visualizing the query you just made.**
# 
# This is where the BQ and Kernels integration really shines. Before, you would have to query and download the output in BigQuery, then re-upload the data into a local Integrated Development Environment (IDE), configure related settings, and then do your analysis. Now, with just a few clicks, you can do all of that in the same place right here in Kernels!

#  ## Step 5: Create a Model using BigQuery ML
# But wait, there's more! In addition to forming standard SQL queries on your data, BigQuery also has an extension called BigQuery ML (BQML) which enables you to create and execute machine learning models in SQL. If you are more comfortable with SQL than in languages like Python or R, this is a way for you to dip your toes in machine learning in a way you know best. This also further centralizes a data scientist's workflow. Data upload, querying, visualization, feature engineering, creating models, evaluating models, and running predictions can now all be done in SQL and in Kernels using BigQuery.
# 
# Let's create linear regression model aimed to predict the final sales price of a home. Let's train the model on a couple of inputs: GrLivArea, YearBuilt, OverallCond, OverallQual.
# 
# **One common step here before training the model is to use Seaborn to make a correlation matrix. Though not covered in this tutorial, this is commonly used to identify variables with strong correlation to the target variable and would give a good idea of what to use for model training.**

# In[ ]:


# Create a linear model that trains on the variables GrLivArea, YearBuilt, OverallCond, OverallQual.
# GrLivArea = Above grade (ground) living area square feet
# YearBuilt = Year the home was completed
# OverallCond = Overall condition of the home
# OverallQual = Overall quality of the home

model1 = """
          CREATE OR REPLACE MODEL 
            `my-example-dataset.ameshousing.linearmodel`
          OPTIONS(model_type='linear_reg', ls_init_learn_rate=.15, l1_reg=1, max_iterations=5) AS
          SELECT 
            IFNULL(SalePrice, 0) AS label,
            IFNULL(GrLivArea, 0) AS LivingAreaSize,
            YearBuilt, 
            OverallCond, 
            OverallQual
          FROM 
            `my-example-dataset.ameshousing.train`
          """


# In[ ]:


# Set up the query
query_job2 = client.query(model1)

# Make an API request  to run the query and return a pandas DataFrame
linearmodel_GrLivArea = query_job2.to_dataframe()

# See the resulting table made from the query
# print(linearmodel_GrLivArea)


# ## Step 6: Get Training Model Statistics
# Now that we've trained our model, let's get information on how it performed!

# In[ ]:


## Step 5: Create a Model using BigQuery ML
model1_stats = """
         SELECT
           *
         FROM 
           ML.TRAINING_INFO(MODEL `my-example-dataset.ameshousing.linearmodel`)
       """    


# In[ ]:


# Set up the query
query_job3 = client.query(model1_stats)

# Make an API request  to run the query and return a pandas DataFrame
linearmodel_stats = query_job3.to_dataframe()

# See the resulting table made from the query
print(linearmodel_stats)

