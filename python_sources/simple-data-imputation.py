import pandas as pd

# this function imputes values to missing cells to columns of a df 
# by setting the missing value column as "target".
# if there are multiple columns with missing values:
#       - selects a column to work with
#       - fills the missing values in other columns with mean or median
#       - one the column is filled, it proceeds with other columns and apply the same procedure
# utilizes regression models

def imputation_Reg (df, package):

    if not df.isnull().values.any():
        print("No missing value cell to impute data!")
        return df
    
    naColumns = df.columns[df.isna().any()].tolist()
    print('   - Columns ' + str(naColumns) + ' will be filled using ' + package.__class__.__name__)
          
    for c in naColumns:
        
        tempDF = pd.DataFrame(df) # or tempDF = df.copy(deep = True)
        naColumnsOtherThanC = [x for x in naColumns if x != c]
        naColumnsStore = tempDF[naColumnsOtherThanC].values
        tempDF[naColumnsOtherThanC] = tempDF[naColumnsOtherThanC].fillna(df.mean()) # or maybe median
        
        train = tempDF[pd.notnull(tempDF[c])]    
        test = tempDF[pd.isnull(tempDF[c])]
        
        indices = train.index.tolist() + test.index.tolist()
        
        X_train = train.loc[:, train.columns != c]
        y_train = train[c]  
        X_test =  test.loc[:, test.columns != c]       
        y_train = y_train.astype(int)
                
        package.fit(X_train, y_train)
        y_pred = package.predict(X_test)
        test[c] = y_pred
        
        filledColumn = pd.concat([train[c], test[c]], ignore_index=True)
        filledColumnDF = pd.DataFrame({'Indices': indices, 'FilledColumn': filledColumn})
        filledColumnDF = filledColumnDF.sort_values('Indices')
        df[c] = filledColumnDF['FilledColumn'].values
        df[naColumnsOtherThanC] = naColumnsStore
        
    return df