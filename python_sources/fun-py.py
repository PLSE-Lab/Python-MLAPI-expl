def welcome_msg():
    print("SIMPLE FUNCTIONS FOR PRE-PROCESSING")
    print("With some basic functions-- Edit")



# Returns the Cols which has Nulls Values
def data_nullcols(df,i):
    """
    Usage          : Function to get Null Values for each Columns in DataSet
    Implementation : Function takes 2 arguments
                    1. DataFrame (The dataset)
                    2. Integer - 0 : For just printing the output
                               - 1 : For returing the values 
    """                               
    col_lst={}
    j=0
    if(i==0):
        #print("For I ZERO")
        for i in df.columns:
            null_val=df[i].isna().sum()
            if null_val != 0:
                col_lst.update({i:null_val})
                print("Column Name : {0} --> Null Values : {1}".format(i,null_val))
            #else:
                #print("Column Name : {0} has no Null Values".format(i))                    

    else:
        print("Returned the Cols with Null values")
        for i in df.columns:
            null_val=df[i].isna().sum()
            if null_val != 0:
                col_lst.update({i:null_val})
        return col_lst
    
    

# Returns Sample n records from DataSet.
def data_head(df,n):
    return df.head(n)


# Returns number of Rows and Columns of the Dataset 
def  data_shape(df):
    print("Number of Rows    : {0} \nNumber of Columns : {1} ".format(df.shape[0],df.shape[1]))


# Describe Function :
def data_describe(df):
    print("Few Insights of the DataSet")
    return df.describe().T


# Group the cols with datatypes
def data_groupcols(df):
    dataType = df.columns.groupby(df.dtypes)
    #print(dataType)
    #return dataType
    for i,j in dataType.items():
        dd=[]
        [dd.append(a) for a in j]
        print("Data Type: {} --- Columns Names : {}".format(str(i).upper(),str(dd)))
        

# Returns or Drops the Duplicates        
def data_duplicates(df,i):
    if (i==0):
        print("Number of Duplicate Records",df.duplicated().sum())
    else:
        return df.drop_duplicates()
    

# Issue with the function---- Needs to be fixed    
def data_col_unique(df,col):
    return df[col].unique()
    


# Returns number of NA in each cols
def data_isna(df):
    col_lst={}
    j=0
    for i in df.columns:
        null_val=df[i].isna().sum()
        if null_val != 0:
            col_lst.update({i:null_val})
            print("Column Name : {0} --> Null Values : {1}".format(i,null_val))
            j=1
        
    if j==0:
        print("No Columns has NA")
        

# Return Correlation of Target with other Columns        
def data_corr_trg_col(df,col):
    return df.corr()[col].sort_values()  


# Return DataFrame with dropped Columns
def data_drop_cols(df,lst):
    return df.drop(lst,axis=1)


# Return value counts of each Columns
def data_value_counts(df,col):
    return df[col].value_counts()
        
        
    