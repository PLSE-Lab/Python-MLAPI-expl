#Pull exit poll data from CNN for all states with results for the 2016 Presidential Primary.
#Returns an empty data frame in Kaggle docker due to blocked connection, but it works locally.

import requests
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def retrieve(state, party):
    url = 'http://data.cnn.com/ELECTION/2016primary/' + state + '/xpoll/' + party + 'full.json'
    r = requests.get(url).json()
    r = r['polls']
    
    df = pd.io.json.json_normalize(r, ['answers', 'candidateanswers'], 
                                   ['numrespondents', 'question', 'pollname', ['answers', 'answer']]) #,'qid' 
    
    if party == 'D':
        #Clinton and Sanders only
        df = df.query('id in (1746, 1445)')
        #Recode candidate IDs
        df['candidate'] = df['id'].replace({1746: 'Hillary Clinton', 1445: 'Bernie Sanders'}) 
    else:
        #Kasich, Cruz, and Trump only
        df = df.query('id in (36679, 61815, 8639)')
        #Recode candidate IDs
        df['candidate'] = df['id'].replace({36679: 'John Kasich', 61815: 'Ted Cruz', 8639: 'Donald Trump'})  
    
    #Add state column
    df['state'] = state
    
    #Rename and drop a few columns
    df['response'] = df['answers.answer']
    df['sampletotal'] = df['numrespondents']
    del df['answers.answer'], df['id'], df['numrespondents']
    
    return(df)

#Initialize empty dataframe
final = pd.DataFrame()

#List of states with completed primaries
statelist = ['AL', 'AK', 'AZ','AR', 'AS', 'CO', 'DA', 'FL','GA', 'GU', 'HI', 'ID','IL','IA','KY','LA','ME','MA','MI', 'MN','MS','MO','NE','NV','NC','NY','OH','OK','SC','TN','TX','UT','VT','VA','WI','WY','NH']

#For each state and party, retrieve poll data and append to dataframe final
for state in statelist:
    for party in ('D', 'R'):
        try:
            final = final.append(retrieve(state, party))
        except:
            continue

#Rebuild index
final = final.reset_index()
del final['index']

print(final.head(10))

#Export dataframe to csv
#final.to_csv('output_exitpolls.csv')