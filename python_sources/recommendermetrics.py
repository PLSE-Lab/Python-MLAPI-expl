#BUILDING RECOMMENDATION METRICS LIBRARY

#surprise dccumetation : https://surpriselib.com
#you can see the code for metrics at github page in above website

#importing libraries
import itertools
from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
        
    #GetTopN takes in complete list of ratings prediction that come back from some recommender and
    #returns a dictionary that maps user ids to their Top N Ratings.
    #We are using defaultdict object which is simmilar to normal python dictionary  but has 
    #concept of default empty values
    def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if(estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating)) #note parenthesis
        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]
        return topN
        
    #To predict HitRate, we need to pass in both our dictionary of Top N Movies for each user ID and 
    #the set of test movie ratings that were left out of training dataset.
    #We are using Leave One Out Cross Validation to hold back one rating per user and test our ability to 
    #recommend that movie in our Top N lists
    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0
        #for each left out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            #is it in predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
                if (hit):
                    hits += 1
                total += 1
        #compute overall precision
        return (hits/total)
        
    #Cumilative Hit Rate or CHR works exactly the same way as hit rate except now we have rating cutoff value.
    #So, we dont count hit unless predited rating is higher than some threshold
    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0
        #for each left out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            #only look at ability to recommend things the user actually liked...
            if(actualRating >= ratingCutoff):
                #is it in predicted top 10 of this user?
                hit=False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if(int(leftOutMovieID) == int(movieID)):
                        hit = True
                        break
                if(hit):
                    hits += 1
                total += 1
        #compute overall precision
        return (hits/total)
        
        
        
    #Rating Hit Rate (RHR) : Smilar to Hit rate but We keep track of hit rate for each unique rating value
    #So instead of keeping one variable to keep track  of hits and total users, we use another dictionary
    #to keep track of hits and totals of each rating type. Then, we print them all out
    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    
    #ARHR : Simillar to hit rate. Difference uis that we count things up by the reciprocal of ranks of each 
    #hits, inorder to get more credit for hits that occured near the top of Top N list
    #
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total
            
            
            
    # COVERAGE : What percentage of users have at least one "good" recommendation (ABOVE SOME THRESHHOLD)
    #In real world, you would probably have a catalog of items that is larger than the set of items you
    #have recommendations data for and would compute coverage based on that lager dataset intead.
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    #DIVERSITY : To measure diversity, we not only need all of the Top N Recommendations from our system,
    #we need a matrix of similarity scores between every pair of items in our dataset. 
    #Although divesity is easy to explain, coding is little tricky -
    #We start by retreiving similarity matrix(Basically a 2x2 Matrix array that contain similarity scores for every posible
    #combination of items that we can quickly lookup). Then, we go to Top N Recommendations for each user one user at a time.
    #itertools.combinations :
    #This call gives us back every combination of item pairs within the Top N List. Wecanthen iterate through each pair
    #and look at similarity between each pair o items
    #NOTE : SURPRISE maintains INTERNAL IDs for both users and items that are  sequential, and these are different from raw
    #user ids and movie ids that are presentin our actual ratings data.
    #Similarity mmatrix uses those inner user IDs so we need to convert our raw IDs into inner IDs before looking up 
    #similarity scores. We add up all the similarity scores, take the average and subtract it from one to get out
    #our diversity metric
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)



    #NOVELTY : We tak in a handy dictionary of popularity rankings of every item as a parameter, and then, go through
    #every user's top n recommendations and compute avg of all popularity rankings of evey item recommended
    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n

            
            
            
    #NOTE: ABOVE DATA IS HIGHLY COMPUTATIONAL. SAMPLE DATA IN REAL WORLD SCENARIOS
    