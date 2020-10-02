import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

#object that contains dictionaries
class ListChange:
    
    def __init__(self,title,cabin):
        self.title_list=title
        self.cabin_list=cabin

#object used to contains dataframe and model
class DF:
    
    model=None
    
    #constructor object DF
    def __init__(self,path):
        self.df=pd.read_csv(path)
            
    #replace null value of titleCol with a value passed
    def Replace_Null(self,TitleCol,ValueReplace):
        self.df[TitleCol].fillna(ValueReplace,inplace=True)
        
   
    #regex used to get significant word(mr,mrs,miss,...) in Name column
    def create_regex(self,name,t_list):
        for string in t_list:
                regex=re.compile(r"([\s,.]"+string+r"[\s,.])")
                if(regex.search(name)):
                    #print(name + ''+string)
                    return string


    #regex used to get significant word(kind of cabin: A,B,...) in Cabin column
    def create_regex_cabin(self,name,t_list):
        for string in t_list:
            regex=re.compile(r"("+string+r"[1-9])")
            if(regex.search(name)):
                #print(name + ''+string)
                return string  
            
    #replace name get using create_regex function and create different classes
    def replace_gender(self,substring):
        if(substring in [ 'Major', 'Capt', 'Col','Master']):
            return 'Ship_crew'
        elif(substring in ['Don','Mr','Jonkheer','Rev' ]):
            return 'Mr'
        elif(substring in ['Countess', 'Mme']):
            return 'Mrs'
        elif(substring in ['Mlle','Ms']):
            return 'Miss'
        return substring
    
   

    #using one-hot-encoding paradigm to create more column for categorical variables
    def create_dummies_titanic(self,df,TitleCol):              
        title_dummies=pd.get_dummies(self.df[TitleCol])
        self.df=self.df.drop(TitleCol,axis=1)            
        self.df=self.df.join(title_dummies)
    
    #clean data and features engineering 
    def clean_data(self,list_change):
        self.df['Title']=self.df['Name'].map(lambda x: self.create_regex(x,list_change.title_list))
    
        self.df['Cabin']=self.df['Cabin'].map(lambda x: self.create_regex_cabin(x,list_change.cabin_list))

        #replace null value in cabin
        self.Replace_Null('Cabin','U0')
        
        
        self.df['Title']=self.df['Title'].map(lambda x: self.replace_gender(x))
        
        self.create_dummies_titanic(self.df,'Title')
        self.create_dummies_titanic(self.df,'Embarked')
        
        cabin_dummies=pd.get_dummies(self.df['Cabin'])
        #cambio nome variabile per evitare conflitti
        cabin_dummies.columns=['A','B','Ca','D','E','F','G','U0']
        self.df=self.df.drop('Cabin',axis=1)

        self.df=self.df.join(cabin_dummies)
        #elimina variabili meno importanti
        self.df=self.df.drop(['PassengerId','Sex','Name'],axis=1)

    #split age in different range
    def create_range_age(self):
        df_row_null=self.df[self.df['Age'].isnull()]
        df_row_null['Range_age']=np.nan
        self.df=self.df.dropna(how='any',axis=0)
        #trasforma age in range_age di tipo int
        category = pd.cut(self.df.Age,5,labels=[1,2,3,4,5])
        category = category.to_frame()
        category.columns = ['Range_age']
        #concatenate age and its bin
        self.df = pd.concat([self.df,category],axis = 1)
        #converti da tipo category a tipo int
        range_age_int=pd.to_numeric(self.df.Range_age,errors='coerce')
        self.df=self.df.drop('Range_age',axis=1)
        self.df=self.df.join(range_age_int)
        return df_row_null
    
    #find age using a decision tree algorithm
    def find_age(self,df_row_null):
        x=self.df.drop(['Survived','Age','Range_age','Pclass','SibSp','Parch','Dr','A','B','Ca','D','E','F','G','U0','Ticket','Mrs'],axis=1).values
        y=self.df['Range_age'].values

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        tree=DecisionTreeClassifier(criterion="gini")
        tree.fit(x_train,y_train)
        #y_pred_train=tree.predict(x_train)
        #y_pred=tree.predict(x_test)

        #apply model to new df
        features_model=df_row_null.drop(['Survived','Age','Range_age','Pclass','SibSp','Parch','Dr','A','B','Ca','D','E','F','G','U0','Ticket','Mrs'],axis=1);
        p=tree.predict(features_model).tolist()
        df_row_null['Range_age']=p

        self.df=self.df.append(df_row_null)
    
    #find age using a decision tree algorithm     
    def find_age_test(self,df_row_null):
        y=self.df['Range_age'].values
        x=self.df.drop(['Age','Range_age','Pclass','SibSp','Parch','Dr','A','B','Ca','D','E','F','G','U0','Ticket','Mrs'],axis=1).values

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        tree=DecisionTreeClassifier(criterion="gini")
        tree.fit(x_train,y_train)
        #y_pred_train=tree.predict(x_train)
        #y_pred=tree.predict(x_test)

        #accuracy_train=accuracy_score(y_train,y_pred_train)
        #accuracy_test=accuracy_score(y_test,y_pred)

        #apply model to new df
        features_model=df_row_null.drop(['Age','Range_age','Pclass','SibSp','Parch','Dr','A','B','Ca','D','E','F','G','U0','Ticket','Mrs'],axis=1);
        p=tree.predict(features_model).tolist()
        df_row_null['Range_age']=p

        self.df=self.df.append(df_row_null)
    
    #remove variable with less weight for my prediction
    def remove_var(self):
        #rimozione variabili 'inutili'
        self.df=self.df.drop('Ticket',axis=1)
        self.df=self.df.drop('Age',axis=1)
    
    

    #create new variable Family=0 if the passenger is alone
    def create_family(self):
        self.df['Family']=self.df.apply(lambda row: 1 if(row.SibSp+row.Parch>0) else 0,axis=1)
        #elimina colonne inutilizzate
        self.df=self.df.drop('SibSp',axis=1)
        self.df=self.df.drop('Parch',axis=1)
    
    
    

#start pre-processing machine learning on train set
def start_train():
    #train set
    df=DF("../input/train.csv")
    
    #replace null value of cabin with new parameter
    df.Replace_Null('Cabin','U0')
    
    #dictionary of most common title for people
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev','Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess','Don', 'Jonkheer']
    
    #dictionary kind of cabin
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'U0']
     
    #object that contains the two above dictionaries
    list_change=ListChange(title_list,cabin_list)
 
    #start clean data
    df.clean_data(list_change)

    #df with empy age cells
    df_row_null=df.create_range_age()
    
    #predict age
    df.find_age(df_row_null)
    #remove not important variables
    df.remove_var()
    #create new variable family
    df.create_family()
    
    return df
    
def start_test():
    #test set
    df_test=DF("../input/test.csv")
    
    #contains variable survived for test set
    df_s=DF("../input/gender_submission.csv")
    
    #merge two dataframes to create new column Survived on test set
    df_test.df=pd.merge(df_test.df, df_s.df, on =['PassengerId'])
    
    #replace null value of cabin with new parameter
    df_test.Replace_Null('Cabin','U0')
    
    #dictionary of most common title for people
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
            'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
            'Don', 'Jonkheer']
    
    #dictionary kind of cabin
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'U0']
    
    #object that contains the two above dictionaries
    list_change=ListChange(title_list,cabin_list)
    
    #start clean data
    df_test.clean_data(list_change)

    #df with empy age cells
    df_row_null=df_test.create_range_age()
    
    #predict age
    df_test.find_age_test(df_row_null)
    
    #remove not important variables
    df_test.remove_var()
    #create new variable family
    df_test.create_family()

    return df_test



df_train=start_train()

df_test=start_test()



#object that contains start and test set
class DF_Model():
    
    def __init__(self,df_train,_df_test):
        self.df_train=df_train
        self.df_test=df_test

    #create linear regression
    def create_linear_regression(self):
        print('#####Linear Regression#####')
        x_train=self.df_train.df.drop(['Survived','Mr','U0','Pclass','S','G','F','Fare'],axis=1).values
        y_train=self.df_train.df['Survived'].values

        x_test=self.df_test.df.drop(['Survived','Mr','U0','Pclass','S','G','F','Fare'],axis=1).values
        y_test=self.df_test.df['Survived'].values

        ll=LinearRegression()
        ll.fit(x_train,y_train)
        y_pred=ll.predict(x_test)

        print("MSE :" + str(mean_squared_error(y_test,y_pred)))
        print("R2 score : " + str(r2_score(y_test,y_pred)))
        
        ss=StandardScaler()
        x_train_std=ss.fit_transform(x_train)
        x_test_std=ss.transform(x_test)

        ll=LinearRegression()
        ll.fit(x_train_std,y_train)
        y_pred=ll.predict(x_test_std)
        print("MSE Standardized : " + str(mean_squared_error(y_test,y_pred)))
        print("R2 score Standardized : " + str(r2_score(y_test,y_pred)))
        
    #create decision tree and random forest
    def create_random_forest(self):
        print('#####Decision tree#####')
        x_train=self.df_train.df.drop(['Survived','Mr','U0','Pclass','S','G','F','Fare'],axis=1).values
        y_train=self.df_train.df['Survived'].values

        x_test=self.df_test.df.drop(['Survived','Mr','U0','Pclass','S','G','F','Fare'],axis=1).values
        y_test=self.df_test.df['Survived'].values
        
        tree=DecisionTreeClassifier(criterion="gini",max_depth=5)
        tree.fit(x_train,y_train)
        y_pred_train=tree.predict(x_train)
        y_pred_test=tree.predict(x_test)
        self.model=tree
        accuracy_train=accuracy_score(y_train,y_pred_train)
        accuracy_test=accuracy_score(y_test,y_pred_test)
        print("Accuracy train=%.4f test=%.4f"%(accuracy_train,accuracy_test))        
        
        
        #grid-search to split train test and improve accuracy
        print('#####Random forest#####')
        rf = RandomForestRegressor()
        param_grid={
                    'max_depth': [4,5,6,7,10,20],
                    'max_features': [2, 3,4,5,6,7],
                    'n_estimators': [100,200,300]
                    }
        gs=GridSearchCV(estimator=rf, param_grid=param_grid,cv=4)
        gs.fit(x_train,y_train)
        print("Best Score")
        print(gs.best_params_)
        
        print('Best random forest')
        rf_best=RandomForestClassifier(random_state=0,max_features=5,n_estimators=100,max_depth=6)
        rf_best.fit(x_train,y_train)
        y_pred_test=rf_best.predict(x_test)
        print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,y_pred_test))

        print('#####Decision tree grid search#####')
        param_grid={'max_depth':np.arange(3,10)}
        tree=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
        tree.fit(x_train,y_train)
        y_pred_test=tree.predict_proba(x_test)
        print('Best score')
        print(tree.best_params_)
        
        tree=DecisionTreeClassifier(criterion="gini",max_depth=3)
        tree.fit(x_train,y_train)
        y_pred_train=tree.predict(x_train)
        y_pred_test=tree.predict(x_test)
        self.model=tree
        accuracy_train=accuracy_score(y_train,y_pred_train)
        accuracy_test=accuracy_score(y_test,y_pred_test)
        print("Accuracy train=%.4f test=%.4f"%(accuracy_train,accuracy_test))   

    #create logistic regression
    def create_logistic_regression(self):
        print('#####Logistic Regression#####')
        x_train=self.df_train.df.drop(['Survived','Mr','U0','Pclass','S','G','F','Fare'],axis=1).values
        y_train=self.df_train.df['Survived'].values

        x_test=self.df_test.df.drop(['Survived','Mr','U0','Pclass','S','G','F','Fare'],axis=1).values
        y_test=self.df_test.df['Survived'].values

        mms=MinMaxScaler()
        x_train=mms.fit_transform(x_train)
        x_test=mms.transform(x_test)

        #logistic regression recognizes multiclass problem and applied one vs all algorithm
        lr= LogisticRegression()
        lr.fit(x_train,y_train)

        #prediction
        y_pred=lr.predict(x_test)
        #probability
        y_pred_proba=lr.predict_proba(x_test)
        #calculate accuracy

        print("Accuracy " + str(accuracy_score(y_test,y_pred)))
        print("Log Loss " + str(log_loss(y_test,y_pred_proba)))
        
        
df_model=DF_Model(df_train,df_test)
df_model.create_linear_regression()
df_model.create_logistic_regression()
df_model.create_random_forest()