# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import sqlite3
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import collections #for ordered dictionaries
import operator
import math




# Here we use a median ROOKIE profile on how they performed in their first NBA year
median_rookie_profile=np.array([ 10.5698, 1.3730, 3.333, 0.35014, 0.3775, 1.36057,	0.1756,	0.4145,	0.5882, 0.18405, 0.3169, 1.2287, 1.5, \
                              0.75833, 0.333, 0.1293, 0.4686, 0.96771, 3.7127,	-1.3038, 101.8250, 100.0287, 108.6617,	107.3097,	-6.9522,\
                             -7.16339, 0.08238,	0.25833, 11.7294, 0.02383,	0.09719, 0.0637, 8.8123, 0.4125, 0.43329, 0.17217,\
                              0.17928,	101.9604, 104.8519,	87.3778, 0.68857, 0.41982,	0.5091,	1.4833,	3.7295,	3.0334,	2.9912,	10.544,\
                              0.206389,	0.5147,   0.53969,	0.3374,	0.3866,	0.03236, 0.16787, 0.09066,	0.10282, 0.12563, 0.333416,	0.29457, \
                              0.179825, 0.27205,  0,	0.47698,	0.1580,	0.17928, 0.15508, 0.18341,	0.10751, 0.18691, 0.09251,	0.1,\
                              0.102166, 0.13820,  0.1421, 0.11057,	0.12186, 0.12719, 0.06269,	0.10378, 0.17652, 0.10869, 0.149326, 0.489046, \
                              0.261946, 0.139208, 0.2141, 0.5256,	0.29739, 0.12906, 0.216864, 4.28960, 0.84452, 0.768716, 2.04347, 2.987179, \
                              14.5119, 	0.07651,  0,	9.7261,	0.5208,	1.0635,	1.06350, 0.85743, 2.29629, 0.2832, 0.67028,	1.0666,	0.391812,\
                              2.594117,	1.349071, 1.24342,	0.505813,	0,	0.09861, 0.20416, 0.16139,	0.2, 0.33,	0.00633, 0.5,	0.34523, \
                              0.06781, 0.5891])




class MyDatabase:

    def __init__(self, filename):
        self.dbfile=filename  
        self.conn = sqlite3.connect(self.dbfile)

    def closeConnection(self):
        self.conn.commit()
        self.conn.close()
 

def open_database(dbfile):
    dbms = MyDatabase(dbfile)
    return dbms



def DissimilarityMeasure(player_dict, ref_dict):
    #calculate disimilarity between two player using their stat


    #Sort the two dictionaries by key (age)
    player_d = collections.OrderedDict(sorted(player_dict.items()))
    ref_d =    collections.OrderedDict(sorted(ref_dict.items()))



    #weighted SSE error between points
    MatchError=0 
    for x in player_d:

        if x in ref_d:

            weight_player=player_d[x][1]
            weight_ref=ref_d[x][1]

            weight=  min(weight_player,weight_ref)  #average, minimum or other?
            MatchError=MatchError+   np.power((player_d[x][0]-ref_d[x][0]),2)   / weight  


        else:
            MatchError = MatchError+  np.power(player_d[x][0],2)
    
    
    return MatchError 
 


def weighted_sum_objective(w, arg1, arg2):

    Target = arg1.T
    Ref = arg2.T

    w=w.reshape(len(w),1)

    l=0.0 #regularisation
 
    
    return np.sum(np.power((w*Ref- Target), 2)) + l*np.sum(np.power(w, 2))
 



def addPredictionToDatabase(cursor, player_id, pred_season, predicted_stats_list):
    
    #build the sql query for all stats
    
    predicted_stats_list = np.array(predicted_stats_list).astype(str)

    
    #point cursor to Stats table in order to get the column names
    s_rows =  cursor.execute('SELECT * FROM Stats').fetchone()
    
    stat_start=5
    col_names=[]
    col_descriptions=cursor.description
    for i in range(len(s_rows)-stat_start):                    
        #Get the stat name
        col_name=col_descriptions[i+stat_start][0]  
        col_names.append(col_name)

        #first check if stat column names exist in the PredictedStats database. If not then add them
        if cursor.execute('Select * from PredictedStats').fetchone() is None:
            try:
                cursor.execute('ALTER TABLE PredictedStats ADD COLUMN '+col_name+' TEXT')
            except:
                pass
                        

    if len(predicted_stats_list) == len(col_names): #sanity check
        
        sql = 'INSERT OR REPLACE INTO PredictedStats ("player_id", "season",  "%s") VALUES ("%s", "%s", "%s");' % (
                '","'.join(col_names), player_id, pred_season, '","'.join(predicted_stats_list))   

        #Execute query
        try:
            cursor.execute(sql)
            return 1
        except:
            return 0
    else:
        return 0
    


def predictFutureStats(player_id, sorted_Matches, all_player_stats_rows, stat_index, topN):
    
    topN=min(len(sorted_Matches), topN)

    #Get stats of target player    
    p_indx=all_player_stats_rows[:,0]==player_id
    player_stats_rows=all_player_stats_rows[p_indx,:]


    # player_stats_rows =  np.array(stats_cursor.execute('SELECT * FROM Stats where player_id="'+player_id+'"').fetchall())
    player_stat_X=player_stats_rows[:,2].astype(int)
    player_stat_Y=player_stats_rows[:,stat_index].astype(float)


    next_season_age=int(player_stats_rows[-1,:][2])+1


    Ref=np.zeros([len(player_stat_X), topN]) #matrix to hold the statistics of the topN players. we will use this to calculate the prediction weights for the target player
    Ref_weights=np.zeros(topN)
    Ref_next= np.zeros([topN])  #matrix to hold the statistics of the topN players for the next season. We will use this together with the calculated weights to generate the prediction
    #Get stats of topN matched players
    for i in range(topN):

        Ref_weights[i]=sorted_Matches[i][1]
        match_player_id=sorted_Matches[i][0]
        m_indx=all_player_stats_rows[:,0]==match_player_id

        match_player_stats_rows=all_player_stats_rows[m_indx,:]
        match_player_stat_X=match_player_stats_rows[:,2].astype(int)
        match_player_stat_Y=match_player_stats_rows[:,stat_index].astype(float)

        #populate Reference matrix only at x locations (i.e. age) given by target player
        for s in range(len(player_stat_X)):
            loc= np.where(match_player_stat_X==player_stat_X[s])
            if loc[0].size>0:
                Ref[s,i]  = match_player_stat_Y[loc]


        #append the stat from the next season (i.e. to be predicted)
        next_season_match_stat=  match_player_stat_Y[np.where(match_player_stat_X==next_season_age)][0]
        Ref_next[i] = next_season_match_stat


    #Remove any entries in the Ref array where all players have 0 values
    non_zero_indx=[]
    for t in range(Ref.shape[0]):
        if any(Ref[t,:] > 0 ):
            non_zero_indx.append(t)



    # using regression 
    x_train=Ref.T
    y_train=Ref_next
    model =RandomForestRegressor(max_depth=4, random_state=0)
    model.fit(x_train, y_train, sample_weight=Ref_weights)
    predicted_stats=model.predict(player_stat_Y.reshape(1,-1))[0]
    if math.isnan(predicted_stats):
        predicted_stats=0


    return predicted_stats



def calculatePlayersMatch(stat_index, stat_dict, player_stats_rows, all_player_stats_rows, m_player_id):
    ''' find a match between a reference player and a target player   '''

    indx=all_player_stats_rows[:,0]==m_player_id
    match_player_stats_rows=all_player_stats_rows[indx,:]

    #Create dictionary of stat indexed by age
    match_player_stat_dict= {}
    for d in range(len(match_player_stats_rows)):
        weight= int(match_player_stats_rows[d,2])/82 #the weight of the stat is determined by the number of games the player has played in the season. Max games =82  means max weight =1
        match_player_stat_dict[match_player_stats_rows[d,1]] = [float(match_player_stats_rows[d,3]), weight]


    #Ignore any players that we cannot extrapolate from (i.e. their max age is not larger than current age of player of interest)
    next_season_age=str(int(player_stats_rows[-1,:][2])+1)
    if next_season_age in match_player_stat_dict:

        
        #Calculate matching score between player and reference player from database
        MatchError=DissimilarityMeasure(stat_dict, match_player_stat_dict)

    else:

        MatchError=None
    


    return MatchError, m_player_id
 
 

def pool_worker(player_stats_rows, all_players_info_rows, all_player_stats_rows, player_id, stat_index, topN):
    """ calculate the prediction for the parallel pool  """

    num_stats = player_stats_rows.shape[1]

    #Create dictionary of stat indexed by age
    stat_dict= {}

    for d in range(len(player_stats_rows)):
        weight= int(player_stats_rows[d,4])/82 #the weight of the stat is determined by the number of games the player has played in the season. Max games =82  means max weight =1
        stat_dict[player_stats_rows[d,2]] = [float(player_stats_rows[d,stat_index]), weight]
    

    #Loop through ALL other players in the database to find matches
    Matches_dict={}
    
    match_player_single_stat=all_player_stats_rows[:,[0,2,4,stat_index]]
    for id in all_players_info_rows[:,0]:
        MatchError, match_player_id=calculatePlayersMatch(stat_index, stat_dict, player_stats_rows, match_player_single_stat, id) #Only pass the single stat
        
        if MatchError != None:                            
            #Add info to Matches dictionary
            Matches_dict[match_player_id] = MatchError
 
    #Find highest matching players
    Matches_dict = sorted(Matches_dict.items(), key=operator.itemgetter(1))
 
    #Predict next season
    predicted_stats=predictFutureStats(player_id, Matches_dict, all_player_stats_rows, stat_index, topN)
    # print("\tPredicted stat for next season: %f  \t\t\t\t\t "%(predicted_stats) )

    
    return predicted_stats



def getPlayerPrediction(stats_database_file, player_id, test_season, topN):

    #Open database for access   
    stats_dbms = open_database(stats_database_file)   
    stats_cursor= stats_dbms.conn.cursor() 

    #Get info of target player
    player_info_row = np.array(stats_cursor.execute('SELECT * FROM Players where player_id="'+player_id+'"').fetchone())
    player_stats_rows =  np.array(stats_cursor.execute('SELECT * FROM Stats where player_id="'+player_id+'"').fetchall())

    #check if the player+season combination already exists in the Prediction table
    srow=np.array(stats_cursor.execute('SELECT * FROM PredictedStats where player_id="'+player_id+'" and season="'+test_season+'"').fetchone()).astype(float)
    if srow.size == 1:
        
        #Check if the player has any history (ROOKIE or EXPERIENCED player)
        if player_info_row.size==1:
            print("Rookie. ID= %s."%player_id)
            # Here we use a median ROOKIE profile on how they performed in their first NBA year
            prediction=median_rookie_profile

        else:
            print("Predicting player  (",player_id,") ",player_info_row[1]) 
            #check how many past seasons we have in the database
            if player_stats_rows.shape[0]<3:
                #Too few past seasons to calculate a prediction. So our prediction is simply the average of the past two seasons
                print("\t\tLess than 3 seasons. Using previous season stats as the prediction")
                
                if player_stats_rows.shape[0]>1:
                    w1=int(player_stats_rows[-1,4])
                    w2=int(player_stats_rows[-2,4])
                    sw=w1+w2; w1=w1/sw;  w2=w2/sw
                    predicted_stats_list=(w1*player_stats_rows[-1,5:].astype(float)+w2*player_stats_rows[-2,5:].astype(float))
                else:
                    predicted_stats_list=player_stats_rows[-1,5:].astype(float)
            
            else:
                #Calculate the prediction and add it to the database
                print("\t\tRecord does not exist. Matching.")
                
                #Get BASIC info on all other players in database
                all_players_info_rows= np.array(stats_cursor.execute('SELECT * FROM Players where player_id !="'+player_id+'"').fetchall())
                #Get STATS infor on all other players in the database
                all_player_stats_rows= np.array(stats_cursor.execute('SELECT * FROM Stats').fetchall())

                #loop for stats 
                predicted_stats_list=[]
                num_stats=len(player_stats_rows[0])      
                
                for stat_index in range(5,num_stats):
                    result= pool_worker(player_stats_rows, all_players_info_rows, all_player_stats_rows, player_id, stat_index, topN)   
                    predicted_stats_list.append(result)

            prediction = np.array(predicted_stats_list).astype(float) 

        #Add to Predicted Stats database
        addPredictionToDatabase(stats_cursor, player_id, test_season, prediction)


    else:
        print("\t\tPredictions already exist in the database")
        prediction=srow[2:] #skip the first two columns





    #Finish and close database
    stats_dbms.closeConnection()

    return prediction, player_stats_rows


 

def evaluate_prediction_for_Player(test_season, GT_player_info_row, player_stats_rows, predicted_stats_list):
    #calculate average prediction error & baseline error of stats
    Error=[]
    baselineError=[]


    #BASELINE prediction. Just use the data from the last season (if exist)
    if len(player_stats_rows)==0:
        baseline_predicted_stats_list= median_rookie_profile
    else:
        baseline_predicted_stats_list=np.zeros(len(player_stats_rows[0][5:]))
        for i in range(1,len(player_stats_rows)+1):
            if int(test_season)>int(player_stats_rows[-i][1]):
                baseline_predicted_stats_list=player_stats_rows[-i][5:].astype(float)
                break



    for p_stat in range(len(predicted_stats_list)):
        
        #PREDICTION
        #compare with ground truth. Using simple indexing. This will not work well if we are predicting subindices of stats
        GT=max(abs(float(GT_player_info_row[p_stat+6])), 1e-012)
        pcnt_err= min(abs(predicted_stats_list[p_stat]-float(GT_player_info_row[p_stat+6]))  /  GT,5) #capped at 1 (i.e. 500%)
        Error.append(pcnt_err)

        #BASELINE
        #compare with ground truth. Using simple indexing. This will not work well if we are predicting subindices of stats
        b_pcnt_err=min(abs(baseline_predicted_stats_list[p_stat]-float(GT_player_info_row[p_stat+6]))  /  GT,5) #capped at 1 (i.e. 500%)
        baselineError.append(b_pcnt_err)

    Error=np.array(Error)
    baselineError=np.array(baselineError)
    

    #Print weighted error of top 50 features
    feature_index=[19, 24, 25, 13, 18,  0,  3,  1, 36, 10,  9,  4, 11, 41, 8, 20, 30, 87,
                   80, 34, 103, 38, 105, 66, 37, 90, 47, 85, 7, 60, 5, 98, 76, 49, 67, 40,
                   33, 27, 97, 51, 79, 2, 81, 70, 59, 56, 72, 73, 22,  94]

    feature_weights= np.array([0.03962618, 0.0391852 , 0.01784355, 0.01283249, 0.01219561,
                      0.00973923, 0.00890767, 0.00792196, 0.00736699, 0.00710438,
                      0.00706269, 0.00679113, 0.00668696, 0.00638061, 0.00634992,
                      0.00625606, 0.00613168, 0.00599632, 0.00597178, 0.005811  ,
                      0.00580586, 0.00571098, 0.00569257, 0.00539315, 0.00538026,
                      0.00537566, 0.00531788, 0.00515918, 0.00515737, 0.00508881,
                      0.00505557, 0.00496576, 0.00496346, 0.00490503, 0.00488405,
                      0.00486041, 0.00480642, 0.00477501, 0.00471107, 0.00470729,
                      0.00469609, 0.00468946, 0.00465937, 0.00465698, 0.00458559,
                      0.00455966, 0.00454386, 0.0045198 , 0.00449064, 0.0044736 ] )                  
    feature_weights= feature_weights/sum(feature_weights)
    wError          =   np.sum(Error[feature_index]*feature_weights)
    wbaselineError  =   np.sum(baselineError[feature_index]*feature_weights)


    return Error, wError, wbaselineError









db_name='../input/nba-player-stats-19502019/PlayerStatDatabase.db'

#Load ground truth test season database
test_season = '2020'
GT_database_file = '../input/nba-player-stats-19502019/NBA_season2019-20.db'
GT_dbms = open_database(GT_database_file)   
GT_cursor= GT_dbms.conn.cursor()

GT_player_rows = GT_cursor.execute('SELECT * FROM Players ').fetchall()


ErrorDict={}
for p in   range(len(GT_player_rows)):

    print('Player %i of %i'%(p+1,len(GT_player_rows)))
          
        
    player_id=GT_player_rows[p][0]

    #Get ground truth info for target player
    GT_player_info_row = np.array(GT_cursor.execute('SELECT * FROM Players where player_id="'+player_id+'"').fetchone())


    #Calculate the predictions
    topN=1000
    prediction, player_stats_rows=getPlayerPrediction(db_name,   str(player_id), test_season, topN)
 

    #evaluation of predictions for single player    
    _, wError, wbaselineError=evaluate_prediction_for_Player(test_season, GT_player_info_row, player_stats_rows, prediction )          
    ErrorDict[player_id] = [wError, wbaselineError]

Errors=np.array(list(ErrorDict.values()))
AvgErrors=np.nanmean(Errors,axis=0)

#write the error from each of the players to CSV file
np.savetxt("output.csv", Errors[:,0], delimiter=",")

print('Error %f, Baseline error %f'%(AvgErrors[0], AvgErrors[1])) #print the average error and baseline error


#Finish and close databases
GT_dbms.closeConnection()