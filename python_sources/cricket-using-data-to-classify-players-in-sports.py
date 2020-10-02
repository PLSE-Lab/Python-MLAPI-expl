---
title: "Cricket (IPL): Using data to classify players in sports"
author: "Saumya Agarwal asaumya@gmail.com"
date: "2nd June 2018"
output: html_document
---

##Objective
Classify players into different groups using data. Aim is to use unsupervised learning to segments players into different classes by identifying right signals.

Recently I stumbled upon ball by ball data of Indian Premier League (IPL) for last 10 years. Being a cricket and data fan this was a win win for me. Also to understand any machine learning algorithm its best to apply them on fields of your interest.



```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
#Load Libraries
library(dplyr)
library(reshape2)
library(ggplot2)
```


## Understanding the sport

Cricket is a game of bat and ball where 2 teams of 11 players each play. The data which I have is for 20-20 format. In this both teams bat (and ball) for 20 overs each and whoever scores more runs wins.
11 players can generally be classified in 4 different categories 
 
1. Batsman: Objective is to score runs as quickly as possible (4-6 Players in each team)
2. Bowler: Objective is to take wickets and give as less as runs possible (3-5 in each team)
3. All Rounder: Does both batting and bowling (3-5 in each team)
4. Wicket Keeper + Batsman: Stands behind the stumps for fielding team. Objective is to collect the ball and help in dismissing the batsman. Every team needs to have 1 specialist keeper.

Along with that all these players field when a team is bowling where there objective is to stop the ball, take catches and runout

Details of the sports can be looked up at the Wiki page
https://en.wikipedia.org/wiki/Cricket



## Understanding the data

Lets load the deliveries data to see what we have



```{r, echo=F,warning=F}
deliveries_data = read.csv("../input/deliveries.csv",header=T)

summary(deliveries_data)
```

So for each match this table is giving us ball by ball account of everything that happened. Who the batsman was who the bowler was and what happened on that ball eg runs scored , wicket taken , dot ball etc


Lets load the matches data to see what we have

```{r, echo=F,warning=F}
matches_data  = read.csv("../input/matches.csv",header=T)
 
summary(matches_data)
```
This gives us that match level data for each game , which teams played who was winner etc. Also we can see that match id is the key to connect the 2 tables


## Cleaning the data

Data has no missing value so don't need to worry about it. 
I am removing all the matches which were shortened due to rain or other factor. Main reason for the same is I want to see performance of players when we know it was 20 over match.
Also removing super over data which is played if the initial 20 over match is tie.


```{r, echo=F,warning=F}

Dl_matches = matches_data[ matches_data$dl_applied == 1,]

deliveries_data_filter = deliveries_data %>%
        filter(is_super_over == 0 & !(match_id %in% Dl_matches$id)) 
        
```

Now as mentioned above we know what players constitute a team. So to identify them we need to have signals which can describe them. So here are the signals that I am looking at


1. Batting Average: Average runs scored per inning
2. Batting Strike Rate: Runs scored per 100 balls
3. Average Boundaries: Fours and Sixes hit per inning
4. Thirty plus score: Number of innings where batsman score more than 30
5. Not out: Batsman remain not out by end of innings
6. Bowling Average: Average runs given per wicket
7. Bowling Strike Rate: Average balls delivered per wicket
8. Average Dots balled: Average balls per inning where no run was scored
9. Two plus wickets: Number of innings where bowler took 2 or more wickets 
10. Average catches taken: Average number of catches taken by a fielder per inning
11. Average runout done: Average number of run outs done by a fielder per inning
12. Average stumping done: Average number of stumping done by a wicketkeeper per inning (This can only be done by Wicketkeeper)


All these have been defined keeping in mind the objective of each type of player on the ground.

Now there are other signals which we could have added 
Eg: 
Economy rate: Runs given by baller per over but it is a function of Strike Rate and Average.
Singles/ Doubles taken by batsman but with percentage runs in boundary and Avg runs we get the idea of what signals doubles will be

Idea here is to avoid same signals getting captured otherwise when when classifying it will overfit on those features.



Let's create batsman level summary. 
I am removing all such players who have not played 48 balls (8 Overs) across 10 years. This way we ensure not to have noisy data 


```{r, echo=F,warning=F}

# Batsmen level match summary
Batsmen_Match_Level_Summary = deliveries_data_filter %>%
        group_by(match_id,inning,batsman) %>%
        summarize(Runs_Scored = sum(batsman_runs), Balls_Faced = length(ball[wide_runs ==0]), Fours = length(batsman_runs[batsman_runs ==4]), Sixes = length(batsman_runs[batsman_runs ==6])) %>%
        ungroup()

# Batsman dismissal mode
Batsmen_Out_Data = deliveries_data_filter %>%
        filter(player_dismissed != "") 

Batsmen_Out_Data = Batsmen_Out_Data[,c(1,2,19,20)]


colnames(Batsmen_Out_Data)[3] = c("batsman")

# Merging the 2 
Batsmen_Match_Level_Summary = merge(Batsmen_Match_Level_Summary,Batsmen_Out_Data,by=c("match_id","inning","batsman"),all.x=T)
Batsmen_Match_Level_Summary$dismissal_kind = ifelse(is.na(Batsmen_Match_Level_Summary$dismissal_kind),"NotOut",as.character(Batsmen_Match_Level_Summary$dismissal_kind))


# Filter by those who have played atleast 48 balls and create signals

Batsmen_Summary = Batsmen_Match_Level_Summary %>%
        group_by(batsman) %>%
        filter(sum(Balls_Faced) > 47) %>%
        summarize(Total_Innings = length(inning), Avg = sum(Runs_Scored)/Total_Innings, SR = 100*sum(Runs_Scored)/sum(Balls_Faced),  Avg_Fours = sum(Fours)/Total_Innings, Avg_Sixes = sum(Sixes)/Total_Innings, Thirty_Plus = length(Runs_Scored[ Runs_Scored > 29 ])/Total_Innings, Not_Out = length(dismissal_kind[ dismissal_kind %in% c('NotOut','retired hurt')])/Total_Innings) %>%
        ungroup()

summary(Batsmen_Summary)

```

Looking at the summary above we can see that we have 254 players who have been measured on these features.


Next let's go ahead and do the same for Bowlers
Again removing all such cases where bowler has not bowled 8 overs


```{r, echo=F,warning=F}

### Bowler level summary


# Dismissal where bowler gets credit
bowler_dismissal = c("caught","caught and bowled","bowled","lbw","stumped")

# Create match level summary

Bowler_Match_Level_Summary = deliveries_data_filter %>%
        group_by(match_id,inning,bowler) %>%
        summarize(Runs_Given = sum(total_runs), Balls = length(ball) ,  Dots = length(total_runs[total_runs ==0]), Bowler_Wickets = length(dismissal_kind[dismissal_kind %in% bowler_dismissal])) %>%
        ungroup()

# Filetr by those who have bowled atleast 8 overs and create overall bowler summary

Bowler_Summary = Bowler_Match_Level_Summary %>%
        group_by(bowler) %>%
        filter(sum(Balls) > 47) %>%
        summarize(Bowl_Total_Innings = length(inning), Bowl_Avg = sum(Runs_Given)/sum(Bowler_Wickets) , Bowl_SR = sum(Balls)/sum(Bowler_Wickets),  Bowl_Avg_Dots = sum(Dots)/Bowl_Total_Innings, Two_Plus_Wickets = length(Bowler_Wickets[ Bowler_Wickets>1])/Bowl_Total_Innings) %>%
        ungroup()



summary(Bowler_Summary)

```



Looking at the summary above we can see that we have 269 players who have been measured on these features.

Finally lets get the fielding stats



```{r, echo=F,warning=F}

# Fielding data
Fielding_Data = deliveries_data_filter %>%
        mutate(fielder = ifelse(dismissal_kind == bowler_dismissal[2],as.character(bowler),as.character(fielder))) %>%
        filter(fielder != "") %>%
        group_by(fielder) %>%
        summarize(Catches_Taken = length(dismissal_kind[ dismissal_kind %in% bowler_dismissal[1:2]]), Run_Outs_Done = length(dismissal_kind[ dismissal_kind == c("run out")]), Stumped_Done = length(dismissal_kind[ dismissal_kind %in% bowler_dismissal[5]])) %>%
        ungroup()



# Removing Fielding data of substitue fielders as they actually didn;t play those matches

Fielding_Data = subset(Fielding_Data, !grepl("(sub)", Fielding_Data$fielder))

summary(Fielding_Data)

```

Looking at the summary above we can see that we have 409 players who have been measured on these features. Another interesting thing is the spread of Stumping done as compared to other 2. As mentioned above stumping is done by only wicketkeeper we can see that even 3rd quantile of the player has value as 0



Lets club all these together to get player level summary

```{r, echo=F,warning=F}

colnames(Batsmen_Summary)[1] = c("player")
colnames(Bowler_Summary)[1] = c("player")
colnames(Fielding_Data)[1] = c("player")


Player_Summary = merge(Batsmen_Summary,Bowler_Summary,by=c("player"),all.x=T,all.Y=T)
Player_Summary = merge(Player_Summary,Fielding_Data,by=c("player"),all.x=T)

summary(Player_Summary)

```

We see a lot of NAs in bowling stats and some in Fielding stats too. This means that these players don't ball and haven't done any dismissal via fielding as well.
So we can go ahead and replace them with 0. However 2 stats Bowl_SR and Bowl_Avg needs to be handled separately. A bowler needs to have taken a wicket to have some value there. As bowler has not taken any wicket we get infinite value.


The max of Bowling SR among players who have taken at least 1 wicket is `r max(Player_Summary$Bowl_SR[ is.finite(Player_Summary$Bowl_SR) ],na.rm = T)`
The max of Bowling Avg among players who have taken at least 1 wicket is `r max(Player_Summary$Bowl_Avg[ is.finite(Player_Summary$Bowl_Avg) ],na.rm = T)`

So we are replacing them with 100 and 150 respectively. Basically this is not too far of from max of those numbers and good enough to indicate that as bowlers these 2 stats are worse for them.


```{r, echo=F,warning=F}

# Delaing with NAs and infinite values

Player_Summary = Player_Summary %>%
        group_by(player) %>%
        mutate(Bowl_Avg = ifelse(is.infinite(Bowl_Avg) | is.na(Bowl_Avg),150,Bowl_Avg), Bowl_SR  = ifelse(is.infinite(Bowl_SR) | is.na(Bowl_SR),100,Bowl_SR))

Player_Summary[is.na(Player_Summary)]<-0



```

Finally create final columns for the summary and remove redundant columns



```{r, echo=F,warning=F}
Player_Summary = Player_Summary %>%
        group_by(player) %>%
        mutate(Matches_Played = max(Total_Innings,Bowl_Total_Innings), Avg_Boundaries = sum(Avg_Fours,Avg_Sixes),Avg_Catches_Taken = Catches_Taken/Matches_Played, Avg_RunOut_Done = Run_Outs_Done/Matches_Played, Avg_Stumping_Done = Stumped_Done/Matches_Played) %>%
        ungroup()
        

Player_Summary = Player_Summary[ ,c(1,17,3,4,18,7,8,10,11,12,13,19,20,21)]  

```


## Clustering


Now we start with clustering. I am going to use K means clustering for this data. We will first normalize the data and them apply Elbow method to find optimal k. 
Elbow method basically gives Within Group Sum of Squared error (WSSE) for different number of clusters. Ideal k is the one where WSSE is low and adding more k gives us diminishing returns


```{r, echo=F,warning=F}
set.seed(1001)
normalize <- function(x) {
        return ((x - min(x)) / (max(x) - min(x)))
}

Player_Summary_normal = as.data.frame(lapply(Player_Summary[,c(3:14)], normalize))


score_wss_normal <- (nrow(Player_Summary_normal)-1)*sum(apply(Player_Summary_normal,2,var))
for (i in 2:15) score_wss_normal[i] <- sum(kmeans(Player_Summary_normal,
                                                  centers=i)$withinss)

plot(1:15, score_wss_normal[1:15], type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Elbow method to look at optimal clusters for Normalized Data",
     pch=20, cex=2)

diff(score_wss_normal)

```

From the plot above and the difference vector we can see that 5 looks like optimal number but let's look at both 4 and 5

Let apply the K-means on both of them and compare the results



```{r, echo=F,warning=F}

# To normalize and de normalize the data

minvec <- sapply(Player_Summary[,c(3:14)],min)
maxvec <- sapply(Player_Summary[,c(3:14)],max)
denormalize <- function(x,minval,maxval) {
        return(x*(maxval-minval) + minval)
}

set.seed(009)

# Cluster with 5 groups
km_5_cluster_normal = kmeans(Player_Summary_normal,5,nstart = 100)  

km_5_cluster_actual = NULL
test1 = NULL

for(i in 1:length(minvec))
{
        test1 = (km_5_cluster_normal$centers[,i] * (maxvec[i] - minvec[i])) + minvec[i]
        km_5_cluster_actual = cbind(km_5_cluster_actual,test1)
}

# Cluster with 4 groups
km_4_cluster_normal = kmeans(Player_Summary_normal,4,nstart = 100)  

km_4_cluster_actual = NULL
test1 = NULL

for(i in 1:length(minvec))
{
        test1 = (km_4_cluster_normal$centers[,i] * (maxvec[i] - minvec[i])) + minvec[i]
        km_4_cluster_actual = cbind(km_4_cluster_actual,test1)
}


colnames(km_4_cluster_actual) = colnames(Player_Summary[c(3:14)])
colnames(km_5_cluster_actual) = colnames(Player_Summary[c(3:14)])

print("Numbers of players in each cluster for 4 groups is given below")
km_4_cluster_normal$size
print("Numbers of players in each cluster for 5 groups is given below")
km_5_cluster_normal$size


# Adding cluster for each player bacl
Player_Summary_Mapped = cbind(Player_Summary,km_4_cluster_normal$cluster,km_5_cluster_normal$cluster)
Player_Summary_Mapped = Player_Summary_Mapped[,c(1,15,16,2:14)]

colnames(Player_Summary_Mapped)[c(2:3)] = c("Cluster_4_Group","Cluster_5_Group")


print("Distribution of players in the 2 clusters looks as follows")

table(Player_Summary_Mapped$Cluster_4_Group,Player_Summary_Mapped$Cluster_5_Group)

```

So we can see that cluster 1 and 4 of 4-Group cluster has split into 1,2 and 3 of 5-Group cluster. Also both the approach have decent data points in each cluster


Let's look at what are the values basis on which these groups have been made

```{r, echo=F,warning=F}

print("Below are the average value of all the fatures when broken into 4 cluster" )

round(km_4_cluster_actual,2)
```
Remember the objective of each role as mentioned above. Batsman needs to score more and quickly. Bowler needs to take more wickets and bowl economically.


So the groups represent the following 

1. Bowler: Bowl SR and Bowl Avg are good while SR and Avg are low. Along with that they ball a lot of dot balls and have high 2+ Wickets per match
2. Weak Batsman + Wicketkeeper: SR and Avg are low but they have high Stumping and Catching percentage so Wicket keeper are part of this group
3. Batsman + Wicketkeeper: These are best batsman group they have high Avg and SR along with that they score most boundaries and have many 30+ scores as well
4. All Rounder: We can see they have good Avg and SR but worse that that in group 3. Also they have good Bowl SR and Bowl Avg which but again worst that group 1


```{r, echo=F,warning=F}

print("Below are the average value of all the fatures when broken into 5 cluster" )

round(km_5_cluster_actual,2)
```
So the groups represent the following 

1. Bowler: Bowl SR and Bowl Avg are good while SR and Avg are low. Check out how they are better in numbers lower than those in group 1 of 4-Group cluster. This shows that it has further improved good bowler cluster
2. Batting All Rounder: These are All Rounders who have better numbers in batting than bowling
3. Bowling All Rounder:  These are All Rounders who have better numbers in bowling than batting
4. Weak Batsman + Wicketkeeper: Same as above cluster
5. Batsman + Wicketkeeper: Same as above cluster

So basically 5-Group gives us better distinction of All rounder which are batting all rounders and which are bowling. Let's look at the players for each of the groups of 5-Group Cluster.


```{r, echo=F,warning=F}
print("Bowler" )
Player_Summary_Mapped$player[Player_Summary_Mapped$Cluster_5_Group ==1]

print("Batsman + Wicketkeeper" )
Player_Summary_Mapped$player[Player_Summary_Mapped$Cluster_5_Group ==5]

print("Bowling All Rounder" )
Player_Summary_Mapped$player[Player_Summary_Mapped$Cluster_5_Group ==3]

print("Batting All Rounder" )
Player_Summary_Mapped$player[Player_Summary_Mapped$Cluster_5_Group ==2]

print("Weak Batsman + Wicketkeeper" )
Player_Summary_Mapped$player[Player_Summary_Mapped$Cluster_5_Group ==4]


```

From the data and my knowledge of the game it seems we have done decent enough classification. 
In Batting All rounder group we are getting players like R Sharma and C Gayle. they used to bowl regularly in early seasons but for past few seasons they are more know for there batting. So maybe if we give more weightage to recent years cluster would have classified them differently



## Conclusion 

We used the data to help us identify players based on features we can built. In the end we were able to identify players in 5 different categories as mentioned above.
The knowledge of the game was key to building the feature we wanted and this is true for all Machine learning algorithm you work on, to have the right business context.

To improve this model we can give more weightage to recent years or maybe add some other modes like dismissal modes to identify bowler who take same type of wickets or batsmen who tend to get out in similar fashion.
Some things which we can't tell are whether a bowler is spin or fast bowler or which are favourite scoring areas for a batsman because that data is not with us. 

Further this classification can be used to further classify new players.


Let me know your thought via comments section.

