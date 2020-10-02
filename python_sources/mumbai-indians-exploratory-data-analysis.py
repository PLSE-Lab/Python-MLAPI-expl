---
title: "Mumbai Indians - Exploratory Data Analysis"
subtitle: "A sneak peak into Mumbai Indians performance  - 10 years in Indian Premier League"
output:
  html_document:
    number_sections: false
    toc: true
    fig_width: 8
    fig_height: 6
    theme: cosmo
    highlight: tango
    code_folding: hide
---


##**INTRODUCTION**

I have always been fascinated by sports and cricket is one of my favourite games. 
So, I couldnt squander the opportunity to work on the datasets which has all the intricacies of Indian Premier League. 
Being emotionally attached to Mumbai, all the analysis that you see here revolves around Mumbai Indians.
Let's get started.

##**DATA PREPROCESSING**

We have two datasets to work with. 
The first one is matches.csv which has all the details at the match level whereas the second dataset deliveries.csv has all the intrinsic details ball by ball.
As part of this kernel, we wont be doing any detailed analysis at the player level. We will be restricting our analysis at match level with Mumbai Indians being the focal point.
To get the data in our desired format for exploratory data analysis, we will be doing some data wrangling on the above datasets. 
We will be focusing only on Mumbai Indians as the main team. We also will be using acronyms for the teams (e.g. RCB for Royal Challengers Bangalore).
We also would be removing fields on which we wont be doing any analysis. e.g. Refree1, Refree2 etc. wont be part of our final dataset.

The final structure of the dataset that we will come up after the data wrangling is :

1. Season - The year in which the match was played.
2. City - The city in which the match was played.
3. Toss Result - Did Mumbai Indians win the toss. ("W","L")
4. First Actvity - Did Mumbai Indians bat first or field first.("bat","field")
5. Match Result - Did Mumbai Indians win the match.("W","L")
6. Opposition - The team Mumbai Indians was playing against. ("RCB","CSK","DD" etc.)
7. Batsman Runs - 1st to 6th - Runs scored by Mumbai Indians in the power play.
8. Batsman Runs - 7th to 15th - Runs scored by Mumbai Indians in the mid overs.
9. Batsman Runs - 16th to 20th - Runs scored by Mumbai Indians in the slog overs.
10. Extra Runs - 1st to 6th - Extras scored by Mumbai Indians in the power play.
11. Extra Runs - 7th to 15th - Extras scored by Mumbai Indians in the mid overs.
12. Extra Runs - 16th to 20th - Extras scored by Mumbai Indians in the slog overs.
13. Runs Conceeded - 1st to 6th - Runs conceeded by Mumbai Indians in the power play.
14. Runs Conceeded - 7th to 15th - Runs conceeded by Mumbai Indians in the mid overs.
15. Runs Conceeded - 16th to 20th - Runs conceeded by Mumbai Indians in the slog overs.
16. Extra Runs Conceeded - 1st to 6th - Extras conceeded by Mumbai Indians in the power play.
17. Extra Runs Conceeded - 7th to 15th - Extras conceeded by Mumbai Indians in the mid overs.
18. Extra Runs Conceeded - 16th to 20th - Extras conceeded by Mumbai Indians in the slog overs.
19. Wickets Taken - 1st to 6th - Wickets taken by Mumbai Indians in the power play.
20. Wickets Taken - 7th to 15th - Wickets taken by Mumbai Indians in the mid overs.
21. Wickets Taken - 16th to 20th - Wickets taken by Mumbai Indians in the slog overs.
22. Wickets Lost - 1st to 6th - Wickets lost by Mumbai Indians in the power play.
23. Wickets Lost - 7th to 15th - Wickets lost by Mumbai Indians in the mid overs.
24. Wickets Lost - 16th to 20th - Wickets lost by Mumbai Indians in the slog overs.


We will be mainly using dplyr for data wrangling and ggplot for visuals.

```{r}

# Importing the necessary libraries
library(tidyverse)
library(gridExtra)

#Importing the dataset
dataset_matches <- read.csv('../input/matches.csv')
dataset_deliveries <- read.csv('../input/deliveries.csv')

# Picking up records when Mumbai Indians had to bat first
dataset_MI_bat <- dataset_deliveries %>%
  filter(batting_team == "Mumbai Indians") %>%
  mutate(wickets_fell = if_else(player_dismissed == "",0,1)) %>%
  select(match_id, bowling_team, over, ball, batsman_runs, extra_runs, wickets_fell) %>%
  group_by(match_id, bowling_team, over) %>%
  summarise(batsmen_runs = sum(batsman_runs), extra_runs = sum(extra_runs), wickets = sum(wickets_fell)) %>%
  mutate(over_category = if_else(over <= 6,"1st to 6th",if_else(over > 6 & over <=15, "7th to 15th","16th to 20th"))) %>%
  group_by(match_id, bowling_team, over_category) %>%
  summarise(batsman_runs = sum(batsmen_runs), extra_runs = sum(extra_runs), wickets = sum(wickets))
  
 #Splitting the runs scored, extras scored and wickets lost in three different datasets.
  dataset_MI_bat_1 <- dataset_MI_bat %>%
  select(match_id, bowling_team, over_category, batsman_runs)
  
dataset_MI_bat_2 <- dataset_MI_bat %>%
  select(match_id, bowling_team, over_category, extra_runs)

dataset_MI_bat_3 <- dataset_MI_bat %>%
  select(match_id, bowling_team, over_category, wickets)
  
  
#Converting the rows into columns.
spread_1 <- dataset_MI_bat_1 %>%
  spread(over_category, batsman_runs) %>%
  plyr::rename(c("1st to 6th" = "Batsman Runs - 1st to 6th","7th to 15th" = "Batsman Runs - 7th to 15th","16th to 20th" = "Batsman Runs - 16th to 20th"))

spread_2 <- dataset_MI_bat_2 %>%
  spread(over_category, extra_runs) %>%
  plyr::rename(c("1st to 6th" = "Extra Runs - 1st to 6th","7th to 15th" = "Extra Runs - 7th to 15th","16th to 20th" = "Extra Runs - 16th to 20th"))

spread_3 <- dataset_MI_bat_3 %>%
  spread(over_category, wickets) %>%
  plyr::rename(c("1st to 6th" = "Wickets Lost - 1st to 6th","7th to 15th" = "Wickets Lost - 7th to 15th","16th to 20th" = "Wickets Lost - 16th to 20th"))

#Joining all the three datasets back together
dataset_MI_bat <- spread_1 %>%
  inner_join(spread_2, by = c('match_id','bowling_team')) %>%
  inner_join(spread_3, by = c('match_id','bowling_team')) %>%
  select("match_id","bowling_team","Batsman Runs - 1st to 6th", "Batsman Runs - 7th to 15th","Batsman Runs - 16th to 20th","Extra Runs - 1st to 6th","Extra Runs - 7th to 15th","Extra Runs - 16th to 20th","Wickets Lost - 1st to 6th","Wickets Lost - 7th to 15th","Wickets Lost - 16th to 20th") 

#Removing the variables which are no longer required.
rm(dataset_MI_bat_1)
rm(dataset_MI_bat_2)
rm(dataset_MI_bat_3)
rm(spread_1)
rm(spread_2)
rm(spread_3)


# Picking up records when Mumbai Indians had to field first
dataset_MI_bowl <- dataset_deliveries %>%
  filter(bowling_team == "Mumbai Indians") %>%
  mutate(wickets_fell = if_else(player_dismissed == "",0,1)) %>%
  select(match_id, batting_team, over, ball, batsman_runs, extra_runs, wickets_fell) %>%
  group_by(match_id, batting_team, over) %>%
  summarise(batsmen_runs = sum(batsman_runs), extra_runs = sum(extra_runs), wickets = sum(wickets_fell)) %>%
  mutate(over_category = if_else(over <= 6,"1st to 6th",if_else(over > 6 & over <=15, "7th to 15th","16th to 20th"))) %>%
  group_by(match_id, batting_team, over_category) %>%
  summarise(batsman_runs = sum(batsmen_runs), extra_runs = sum(extra_runs), wickets = sum(wickets))

 #Splitting the runs conceeded, extras conceeded and wickets taken in three different datasets.
  dataset_MI_bowl_1 <- dataset_MI_bowl %>%
  select(match_id, batting_team, over_category, batsman_runs)
  
dataset_MI_bowl_2 <- dataset_MI_bowl %>%
  select(match_id, batting_team, over_category, extra_runs)

dataset_MI_bowl_3 <- dataset_MI_bowl %>%
  select(match_id, batting_team, over_category, wickets)

#Converting rows into columns
spread_1 <- dataset_MI_bowl_1 %>%
  spread(over_category, batsman_runs) %>%
  plyr::rename(c("1st to 6th" = "Runs Conceeded - 1st to 6th","7th to 15th" = "Runs Conceeded - 7th to 15th","16th to 20th" = "Runs Conceeded - 16th to 20th"))

spread_2 <- dataset_MI_bowl_2 %>%
  spread(over_category, extra_runs) %>%
  plyr::rename(c("1st to 6th" = "Extra Runs Conceeded - 1st to 6th","7th to 15th" = "Extra Runs Conceeded - 7th to 15th","16th to 20th" = "Extra Runs Conceeded - 16th to 20th"))

spread_3 <- dataset_MI_bowl_3 %>%
  spread(over_category, wickets) %>%
  plyr::rename(c("1st to 6th" = "Wickets Taken - 1st to 6th","7th to 15th" = "Wickets Taken - 7th to 15th","16th to 20th" = "Wickets Taken - 16th to 20th"))

#Joining all the three datasets back together
dataset_MI_bowl <- spread_1 %>%
  inner_join(spread_2, by = c('match_id','batting_team')) %>%
  inner_join(spread_3, by = c('match_id','batting_team')) %>%
  select("match_id","batting_team","Runs Conceeded - 1st to 6th", "Runs Conceeded - 7th to 15th","Runs Conceeded - 16th to 20th","Extra Runs Conceeded - 1st to 6th","Extra Runs Conceeded - 7th to 15th","Extra Runs Conceeded - 16th to 20th","Wickets Taken - 1st to 6th","Wickets Taken - 7th to 15th","Wickets Taken - 16th to 20th")

#Removing variables which no longer will be used.
rm(dataset_MI_bowl_1)
rm(dataset_MI_bowl_2)
rm(dataset_MI_bowl_3)
rm(spread_1)
rm(spread_2)
rm(spread_3)

# Replacing the Nulls with 0
dataset_MI_bat <- dataset_MI_bat %>% replace(., is.na(.), 0)
dataset_MI_bowl <- dataset_MI_bowl %>% replace(., is.na(.), 0)

#Joining the batting first and bowl first dataset.
dataset_detail <- dataset_MI_bowl %>%
  inner_join(dataset_MI_bat, by = 'match_id')

#Picking up only the necessary columns and renaming it
dataset_detail <- dataset_detail %>%
  select("match_id","batting_team","Runs Conceeded - 1st to 6th", "Runs Conceeded - 7th to 15th","Runs Conceeded - 16th to 20th","Extra Runs Conceeded - 1st to 6th","Extra Runs Conceeded - 7th to 15th","Extra Runs Conceeded - 16th to 20th","Wickets Taken - 1st to 6th","Wickets Taken - 7th to 15th","Wickets Taken - 16th to 20th","Batsman Runs - 1st to 6th", "Batsman Runs - 7th to 15th","Batsman Runs - 16th to 20th","Extra Runs - 1st to 6th","Extra Runs - 7th to 15th","Extra Runs - 16th to 20th","Wickets Lost - 1st to 6th","Wickets Lost - 7th to 15th","Wickets Lost - 16th to 20th") %>%
  plyr::rename(c("batting_team" = "Opposition"))

#Removing variables which no longer will be used.
rm(dataset_deliveries)
rm(dataset_MI_bat)
rm(dataset_MI_bowl)

# Taking Mumbai Indians dataset from matches.csv and merging it with the main dataset which was preprocessed above.
bat_first <- dataset_matches %>%
  filter(team1 == "Mumbai Indians") %>%
  select(id, season, city, toss_winner, toss_decision, winner, win_by_runs, win_by_wickets, venue, umpire1, umpire2) %>%
  mutate(match_result = if_else(winner != "Mumbai Indians","L","W"),
         toss_result = if_else(toss_winner != "Mumbai Indians","L","W"),
         first_activity = if_else((toss_result == "L" & toss_decision == "field") | (toss_result == "W" & toss_decision == "bat"), "bat","field")) %>%
  select(id, season, city, venue,umpire1, umpire2, toss_result, first_activity, match_result, win_by_runs, win_by_wickets ) %>%
  plyr::rename(c("id" = "match_id"))

field_first <- dataset_matches %>%
  filter(team2 == "Mumbai Indians") %>%
  select(id, season, city, toss_winner, toss_decision, winner, win_by_runs, win_by_wickets, venue, umpire1, umpire2) %>%
  mutate(match_result = if_else(winner != "Mumbai Indians","L","W"),
         toss_result = if_else(toss_winner != "Mumbai Indians","L","W"),
         first_activity = if_else((toss_result == "L" & toss_decision == "field") | (toss_result == "W" & toss_decision == "bat"), "bat","field")) %>%
  select(id, season, city, venue,umpire1, umpire2, toss_result, first_activity, match_result, win_by_runs, win_by_wickets ) %>%
  plyr::rename(c("id" = "match_id"))

bat_first <- bat_first %>%
  inner_join(dataset_detail, by = 'match_id')

field_first <- field_first %>%
  inner_join(dataset_detail, by = 'match_id')

final_dataset <- bat_first %>% rbind(field_first)

#Matches played in Dubai had city information missing. Filling in the missing data.
temp_dataset_1 <- final_dataset %>% 
  filter(city != "")

temp_dataset_2 <- final_dataset %>% 
  filter(city == "") %>% replace("city","Dubai")

final_dataset <- temp_dataset_1 %>% rbind(temp_dataset_2)

# Removing the variables which will not be used further.
rm(bat_first)
rm(temp_dataset_1)
rm(temp_dataset_2)
rm(field_first)
rm(dataset_detail)
rm(dataset_matches)

final_dataset <- final_dataset[-1]

# Removing results which ended up in a tie as we wont be analysing those.
final_dataset <- final_dataset %>%
  mutate(MatchResult = if_else(win_by_runs == 0 & win_by_wickets == 0, "Tie","B")) %>%
  filter (MatchResult != "Tie") %>%
  select(-MatchResult)

#Rising Pune Supergiants was spelled in two different ways. Correcting the data.
final_dataset <- final_dataset %>%
  mutate(Opposition = if_else(Opposition == "Rising Pune Supergiant","Rising Pune Supergiants",as.character(Opposition)))

#Adding acronyms to team names.
final_dataset <- final_dataset %>%
  mutate(Opposition = if_else(Opposition == "Chennai Super Kings","CSK",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Deccan Chargers","DC",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Delhi Daredevils","DD",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Gujarat Lions","GL",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Kings XI Punjab","KXIP",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Kochi Tuskers Kerala","KTK",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Kolkata Knight Riders","KKR",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Pune Warriors","PW",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Rajasthan Royals","RR",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Rising Pune Supergiants","RPS",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Royal Challengers Bangalore","RCB",as.character(Opposition))
         ,Opposition = if_else(Opposition == "Sunrisers Hyderabad","SRH",as.character(Opposition))
         )

#Getting the final dataset ready with the necessary columns.         
final_dataset <- final_dataset %>%
  select(-3:-5)
```

##**EXPLORATORY DATA ANALYSIS**

### **1. Win percentage by Season**

Lets have a look at their performance each year from 2008 to 2017.

```{r}
blank_theme <- theme_minimal()+
  theme(
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  panel.border = element_blank(),
  panel.grid=element_blank(),
  axis.ticks = element_blank(),
  plot.title=element_text(size=14, face="bold")
  )

plot1 <- final_dataset %>%
  filter(season == 2008) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2008") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot2 <- final_dataset %>%
  filter(season == 2009) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2009") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot3 <- final_dataset %>%
  filter(season == 2010) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2010") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot4 <- final_dataset %>%
  filter(season == 2011) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2011") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot5 <- final_dataset %>%
  filter(season == 2012) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2012") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot6 <- final_dataset %>%
  filter(season == 2013) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2013") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot7 <- final_dataset %>%
  filter(season == 2014) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2014") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot8 <- final_dataset %>%
  filter(season == 2015) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2015") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot9 <- final_dataset %>%
  filter(season == 2016) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2016") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot10 <- final_dataset %>%
  filter(season == 2017) %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Season 2017") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10, nrow = 5, ncol = 2)

```

**Observation : **
2010, 2011, 2013, 2015 and 2017 have been the best seasons for Mumbai Indians where they have won more than 60% of their matches.
2008 and 2016 have been not so good wherein they have won only 50% of their matches whereas season 2009 and 2014 have been seasons to forget.


###**2. Win percentage against Teams**

Lets see how their performances have been against different teams (from 2008 to 2017).
We will be only looking at teams against which Mumbai Indians have played more than 10 matches and which are currently playing in IPL.

The teams are :

1. Chennai Super Kings
2. Delhi Daredevils
3. Kolkata Knight Riders
4. Kings XI Punjab
5. Royal Challengers Bangalore
6. Sunrisers Hyderabad

```{r}
plot1 <- final_dataset %>%
  filter(Opposition == "CSK") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Chennai Super Kings") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot2 <- final_dataset %>%
  filter(Opposition == "DD") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Delhi Daredevils") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot3 <- final_dataset %>%
  filter(Opposition == "KKR") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Kolkata Knight Riders") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot4 <- final_dataset %>%
  filter(Opposition == "KXIP") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Kings XI Punjab") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot5 <- final_dataset %>%
  filter(Opposition == "RCB") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Royal Challengers Bangalore") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot6 <- final_dataset %>%
  filter(Opposition == "SRH") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Sunrisers Hyderabad") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, nrow = 3, ncol = 2)

```

**Observation : **
Mumbai Indians seem to perform exceptionally well against Kolkata Kinight Riders winning 76% of their matches.
Their performances have also been good against Chennai Super Kings, Royal Challengers Bangalore and Delhi Daredevils.
Kings XI Punjab and Sunrisers Hyderabad are the two teams against which Mumbai struggles to maintain their high winning ratio.


###**3. Win percentage based on first activity in the match**
```{r}
plot1 <- final_dataset %>%
  filter(first_activity == "bat") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Bat First") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot2 <- final_dataset %>%
  filter(first_activity == "field") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Field First") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  


grid.arrange(plot1, plot2, ncol = 2)

```

**Observation : **
We dont see any significant difference in the win percentage when Mumbai Indians bat or field first.

###**4. Win percentage based on toss result**
```{r}
plot1 <- final_dataset %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Toss Won") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot2 <- final_dataset %>%
  filter(toss_result == "L") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Toss Lost") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  


grid.arrange(plot1, plot2, ncol = 2)
```

**Observation : **
Losing the toss seems to be a good omen as MI have won 60% of the matches wherein they have lost the toss as compared to 56% when they have won it.


###**5. Win percentage in different cities and toss result**

We will only consider the home cities for the above teams and Mumbai Indians.
```{r}
final_dataset %>%
  filter(city %in% c("Mumbai","Kolkata","Bangalore","Delhi","Hyderabad","Chandigarh","Chennai")) %>%
  group_by(city) %>%
  summarise(win_percentage = round((sum(if_else(match_result == "W",1,0))/n()*100),2)) %>%
  ggplot(aes(x = city, y = win_percentage))+ geom_bar(stat = 'identity',fill = '#3182bd') + ggtitle("Overall Win Percentage") + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7,angle = 45, hjust = 1)) +geom_text(aes(label=win_percentage), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25)
```

```{r}
final_dataset %>%
  filter(toss_result == "W") %>%
  filter(city %in% c("Mumbai","Kolkata","Bangalore","Delhi","Hyderabad","Chandigarh","Chennai")) %>%
  group_by(city) %>%
  summarise(win_percentage = round((sum(if_else(match_result == "W",1,0))/n()*100),2)) %>%
  ggplot(aes(x = city, y = win_percentage))+ geom_bar(stat = 'identity',fill = '#3182bd') + ggtitle("Win Percentage after winning the toss") + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7,angle = 45, hjust = 1)) +geom_text(aes(label=win_percentage), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25)
  
```

**Observation : **
Mumbai Indians perform exceptionally well in Bangalore, Kolkata, Hyderabad, Mumbai and Chandigarh.
However, after winning the toss, their win percentage seems to drop in all the cities. The seem to lose all matches in Delhi after winning the toss.
We also see a significant win percentage drop in Hyderabad after winning the toss.

###**6. Win percentage against teams based on toss result**
```{r}
plot1 <- final_dataset %>%
  filter(Opposition == "CSK") %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Chennai Super Kings") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot2 <- final_dataset %>%
  filter(Opposition == "DD") %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Delhi Daredevils") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot3 <- final_dataset %>%
  filter(Opposition == "KKR") %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Kolkata Knight Riders") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot4 <- final_dataset %>%
  filter(Opposition == "KXIP") %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Kings XI Punjab") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot5 <- final_dataset %>%
  filter(Opposition == "RCB") %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Royal Challengers Bangalore") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

plot6 <- final_dataset %>%
  filter(Opposition == "SRH") %>%
  filter(toss_result == "W") %>%
  group_by(match_result) %>%
  summarise(abc = n()) %>%
  arrange(desc(match_result)) %>%
  mutate(cumulative = cumsum(abc),
         midpoint = cumulative - abc / 2,
         label = paste0(round(abc / sum(abc) * 100, 1), "%")) %>%
   ggplot(aes(x = "",y = abc,fill = match_result)) + geom_bar(stat = "identity",width = 1) + coord_polar("y") + ggtitle("Sunrisers Hyderabad") + scale_fill_brewer(palette="Blues")+theme_minimal() + geom_text(aes(x = 1, y = midpoint, label = label), size = 2) + blank_theme + theme(axis.text.x=element_blank(),plot.title = element_text(size =8),legend.text = element_text(size=5), legend.title = element_text(size = 7))  

grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, nrow = 3, ncol = 2)

```

**Observation : **
We observe an increase in win percentage against Chennai Super Kings, Kolkata Knight Riders and Royal Challengers Bangalore when MI win the toss.
However,there is a huge drop in win percentage against Sunrisers Hyderabad, Kings XI Punjab and Delhi Daredevils.

###**7. Win percentage in cities based on the first activity in the match**
```{r}
final_dataset %>%
  filter(city %in% c("Mumbai","Kolkata","Bangalore","Delhi","Hyderabad","Chandigarh","Chennai")) %>%
  group_by(city, first_activity) %>%
  summarise(win_percentage = round((sum(if_else(match_result == "W",1,0))/n()*100),2)) %>%
  ggplot(aes(x = city, y = win_percentage))+ geom_bar(stat = 'identity',fill = '#3182bd') + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7,angle = 45, hjust = 1)) +geom_text(aes(label=win_percentage), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~first_activity)
```

**Observation : **
Mumbai Indians win all their matches in Bangalore and Chennai when they bat first whereas they tend to win all their matches in Hyderabad and Kolkata when they field first.

###**8. Win percentage in cities based on the toss result and first activity on the field**
```{r}
final_dataset %>%
  filter(city %in% c("Mumbai","Kolkata","Bangalore","Delhi","Hyderabad","Chandigarh","Chennai")) %>%
  filter(toss_result == "W") %>%
  group_by(city, first_activity) %>%
  summarise(win_percentage = round((sum(if_else(match_result == "W",1,0))/n()*100),2)) %>%
  ggplot(aes(x = city, y = win_percentage))+ geom_bar(stat = 'identity',fill = '#3182bd') + ggtitle("After winning the toss")  + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7,angle = 45, hjust = 1)) +geom_text(aes(label=win_percentage), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~first_activity)
```

**Observation : **
After winning the toss, Mumbai Indians win all their matches in Chennai batting first whereas they win all their matches in Kolkata fielding first.

###**9. Win percentage against teams based on first activity in the match**
```{r}
final_dataset %>%
  filter(Opposition %in% c("RCB","KKR","DD","SRH","KXIP","CSK")) %>%
  group_by(Opposition, first_activity) %>%
  summarise(win_percentage = round((sum(if_else(match_result == "W",1,0))/n()*100),2)) %>%
  ggplot(aes(x = Opposition, y = win_percentage))+ geom_bar(stat = 'identity',fill = '#3182bd') + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7,angle = 45, hjust = 1)) +geom_text(aes(label=win_percentage), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~first_activity)
  
```

**Observation : **
Mumbai Indians maintain a win percentage of above 55% against all teams except Sunrisers Hyderabad when they bat first whereas their win percentage is quite erratic when they field first.
Their win percentage is above 65% against Kolkata Knight Riders, Royal Challengers Bangalore and Sunrisers and its its below 50% against Chennai Super Kings, Delhi Daredevils and Kings XI Punjab.

###**10. Win percentage against teams based on the toss result and first activity on the field.**
```{r}
final_dataset %>%
  filter(Opposition %in% c("RCB","KKR","DD","SRH","KXIP","CSK")) %>%
  filter(toss_result == "W") %>%
  group_by(Opposition, first_activity) %>%
  summarise(win_percentage = round((sum(if_else(match_result == "W",1,0))/n()*100),2)) %>%
  ggplot(aes(x = Opposition, y = win_percentage))+ geom_bar(stat = 'identity',fill = '#3182bd') + ggtitle("After winning the toss") + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=win_percentage), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25, hjust = 1.25) + coord_flip() + facet_wrap(~first_activity)
  
```

**Observation : **
They seem to maintain similar winning percentage against Sunrisers Hyderabad, Kolkata Knight Riders and Chennai Super Kings irrespective of whether they field first or bat first after winning the toss.
However, the seem to lose all their matches against Delhi Daredevils when they win the toss and chose to field first.

###**11. Run spread when batting first**
```{r}
final_dataset %>%
  filter(first_activity == "bat", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(total_runs_scored = `Batsman Runs - 1st to 6th` + `Extra Runs - 1st to 6th` + `Batsman Runs - 7th to 15th`+ `Batsman Runs - 16th to 20th` + `Extra Runs - 7th to 15th` + `Extra Runs - 16th to 20th`) %>%
  ggplot(aes(x = match_result, y = total_runs_scored,fill = match_result))  + scale_fill_brewer(palette="Blues") + geom_boxplot()
```

**Observation : **
Mumbai Indians have scored 175 runs on an average when they have won the matches whereas this average drops to 150 for matches they have lost.
Also, there is no overlap in the IQR when we compare the two match results.

###**12. Run spread batting first against teams**
```{r}
final_dataset %>%
  filter(first_activity == "bat", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(total_runs_scored = `Batsman Runs - 1st to 6th` + `Extra Runs - 1st to 6th` + `Batsman Runs - 7th to 15th`+ `Batsman Runs - 16th to 20th` + `Extra Runs - 7th to 15th` + `Extra Runs - 16th to 20th`) %>%
  ggplot(aes(x = match_result, y = total_runs_scored,fill = match_result))  + scale_fill_brewer(palette="Blues") + geom_boxplot() + facet_wrap(~Opposition)
```

**Observation : **
Looking at the size of the boxes for the matches where they have won, Mumbai Indians show some decent consistency. 
The boxes are quite compact for 4 out of the 6 teams where they have consistently scored approx. 180 runs.

###**13. Run spread when batting second**
```{r}
final_dataset %>%
  filter(first_activity == "field", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(total_runs_scored = `Batsman Runs - 1st to 6th` + `Extra Runs - 1st to 6th` + `Batsman Runs - 7th to 15th`+ `Batsman Runs - 16th to 20th` + `Extra Runs - 7th to 15th` + `Extra Runs - 16th to 20th`) %>%
  ggplot(aes(x = match_result, y = total_runs_scored,fill = match_result))  + scale_fill_brewer(palette="Blues") + geom_boxplot()
```

**Observation : **
The averages between matches where they win or lose just differ by approx 10 runs when Mumbai Indians bat second.
Their average winning score when they bat second is 160 runs.

###**14. Run spread batting second against teams.**
```{r}
final_dataset %>%
  filter(first_activity == "field", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(total_runs_scored = `Batsman Runs - 1st to 6th` + `Extra Runs - 1st to 6th` + `Batsman Runs - 7th to 15th`+ `Batsman Runs - 16th to 20th` + `Extra Runs - 7th to 15th` + `Extra Runs - 16th to 20th`) %>%
  ggplot(aes(x = match_result, y = total_runs_scored,fill = match_result))  + scale_fill_brewer(palette="Blues") + geom_boxplot() + facet_wrap(~Opposition)
```

**Observation : **
In we look at the box plot for Delhi Daredevils what we see is something strange.
This shows that, Mumbai Indians are able to chase down low scores against Delhi. 155 is the highest total Mumbai Indians have chased down against Delhi.
Anything above that and Mumbai tend to lose the match.

###**15. Run Rate by Over Comparison - Batting First**
```{r}
final_dataset %>%
  filter(first_activity == "bat", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(runs_scored_power_play = `Batsman Runs - 1st to 6th` + `Extra Runs - 1st to 6th`
         ,runs_scored_mid_overs = `Batsman Runs - 7th to 15th` + `Extra Runs - 7th to 15th`
         ,runs_scored_slog_overs = `Batsman Runs - 16th to 20th` + `Extra Runs - 16th to 20th`) %>%
  select(Opposition, runs_scored_power_play, runs_scored_mid_overs, runs_scored_slog_overs) %>%
  group_by(Opposition) %>%
  summarise(`Power Play` = mean(runs_scored_power_play)/6
            ,`Mid Overs` = mean(runs_scored_mid_overs)/9
            ,`Slog Overs` = mean(runs_scored_slog_overs)/5) %>%
  gather(Over_Category, Run_Rate, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Run_Rate)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 12)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Run_Rate,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
```

**Observation : **
The increasing pattern of the bargraphs in general show that Mumbai Indians generally start slowly and finish strong.
Mumbai Indians have a very good scoring rate in Slog overs against all teams except Sunrisers Hyderabad

###**16. Run Rate by Over Comparison - Batting Second**
```{r}
final_dataset %>%
  filter(first_activity == "field", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(runs_scored_power_play = `Batsman Runs - 1st to 6th` + `Extra Runs - 1st to 6th`
         ,runs_scored_mid_overs = `Batsman Runs - 7th to 15th` + `Extra Runs - 7th to 15th`
         ,runs_scored_slog_overs = `Batsman Runs - 16th to 20th` + `Extra Runs - 16th to 20th`) %>%
  select(Opposition, runs_scored_power_play, runs_scored_mid_overs, runs_scored_slog_overs) %>%
  group_by(Opposition) %>%
  summarise(`Power Play` = mean(runs_scored_power_play)/6
            ,`Mid Overs` = mean(runs_scored_mid_overs)/9
            ,`Slog Overs` = mean(runs_scored_slog_overs)/5) %>%
  gather(Over_Category, Run_Rate, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Run_Rate)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 10)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Run_Rate,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
```

**Observation : **
There is no specific pattern that we observe here as far as trend is concerned. 
But they seem to start strong with a high run rate in the power play. This actually makes sense as they are chasing down a total.
However, they seem to struggle early on against Royal Challengers Bangalore and Sunrisers Hyderabad.

###**17. Economy Rate by Over Comparison - Fielding First**
```{r}
final_dataset %>%
  filter(first_activity == "field", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(runs_conceeded_power_play = `Runs Conceeded - 1st to 6th` + `Extra Runs Conceeded - 1st to 6th`
         ,runs_conceeded_mid_overs = `Runs Conceeded - 7th to 15th` + `Extra Runs Conceeded - 7th to 15th`
         ,runs_conceeded_slog_overs = `Runs Conceeded - 16th to 20th` + `Extra Runs Conceeded - 16th to 20th`) %>%
  select(Opposition, runs_conceeded_power_play, runs_conceeded_mid_overs, runs_conceeded_slog_overs) %>%
  group_by(Opposition) %>%
   summarise(`Power Play` = mean(runs_conceeded_power_play)/6
            ,`Mid Overs` = mean(runs_conceeded_mid_overs)/9
            ,`Slog Overs` = mean(runs_conceeded_slog_overs)/5) %>%
  gather(Over_Category, Economy_Rate, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Economy_Rate)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 12)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Economy_Rate,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
  
```

**Observation : **
We again see an increasing trend where they have a better economy rate in the power plays as compared to other two segments.
They also tend to give a lot of runs in the slog overs against Chennai Super Kings and Royal Challengers Bangalore.

###**18. Economy Rate by Over Comparison - Fielding Second**
```{r}
final_dataset %>%
  filter(first_activity == "bat", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  mutate(runs_conceeded_power_play = `Runs Conceeded - 1st to 6th` + `Extra Runs Conceeded - 1st to 6th`
         ,runs_conceeded_mid_overs = `Runs Conceeded - 7th to 15th` + `Extra Runs Conceeded - 7th to 15th`
         ,runs_conceeded_slog_overs = `Runs Conceeded - 16th to 20th` + `Extra Runs Conceeded - 16th to 20th`) %>%
  select(Opposition, runs_conceeded_power_play, runs_conceeded_mid_overs, runs_conceeded_slog_overs) %>%
  group_by(Opposition) %>%
 summarise(`Power Play` = mean(runs_conceeded_power_play)/6
            ,`Mid Overs` = mean(runs_conceeded_mid_overs)/9
            ,`Slog Overs` = mean(runs_conceeded_slog_overs)/5) %>%
  gather(Over_Category, Economy_Rate, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Economy_Rate)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 10)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Economy_Rate,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
  
```

**Observation : **
The have the worst economy rate against Chennai Super Kings during the Power Plays.
If we talk about the mid overs, their economy rate is almost the same against all teams.
They have a very good economy rate against Delhi Daredevils and Sunrisers Hyderabad in the final slog overs.

###**19. Average Wickets Lost - Batting First**
```{r}
final_dataset %>%
  filter(first_activity == "bat", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  select(Opposition, `Wickets Lost - 1st to 6th`, `Wickets Lost - 7th to 15th`, `Wickets Lost - 16th to 20th`) %>%
  group_by(Opposition) %>%
  summarise(`Power Play` = mean(`Wickets Lost - 1st to 6th`)
            ,`Mid Overs` = mean(`Wickets Lost - 7th to 15th`)
            ,`Slog Overs` = mean(`Wickets Lost - 16th to 20th`)) %>%
  gather(Over_Category, Avg_Wickets_Lost, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Avg_Wickets_Lost)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 5)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Avg_Wickets_Lost,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
```

**Observation : **
They seem to lose 1.7 wickets on an average against Royal Challengers Bangalore and Sunrisers Hyderabad in the Power Play overs.
They also tend to lose more wickets in the slog overs against all team. This makes sense as they are trying to up the run rate to finish the innings strong.


###**20. Average Wickets Lost - Batting Second**
```{r}
final_dataset %>%
  filter(first_activity == "field", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  select(Opposition, `Wickets Lost - 1st to 6th`, `Wickets Lost - 7th to 15th`, `Wickets Lost - 16th to 20th`) %>%
  group_by(Opposition) %>%
  summarise(`Power Play` = mean(`Wickets Lost - 1st to 6th`)
            ,`Mid Overs` = mean(`Wickets Lost - 7th to 15th`)
            ,`Slog Overs` = mean(`Wickets Lost - 16th to 20th`)) %>%
  gather(Over_Category, Avg_Wickets_Lost, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Avg_Wickets_Lost)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 5)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Avg_Wickets_Lost,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
```

**Observation : **
An interesting observation here is that they tend to lose more wickets on an average in the mid overs (except against Delhi and Sunrisers Hyderabad).
They seem to start cautiously against Chennai where they just lose 0.7 wickets on an average in Power Plays.


###**21. Average Wickets Taken - Fielding First**
```{r}
final_dataset %>%
  filter(first_activity == "field", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  select(Opposition, `Wickets Taken - 1st to 6th`, `Wickets Taken - 7th to 15th`, `Wickets Taken - 16th to 20th`) %>%
  group_by(Opposition) %>%
  summarise(`Power Play` = mean(`Wickets Taken - 1st to 6th`)
            ,`Mid Overs` = mean(`Wickets Taken - 7th to 15th`)
            ,`Slog Overs` = mean(`Wickets Taken - 16th to 20th`)) %>%
  gather(Over_Category, Avg_Wickets_Taken, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Avg_Wickets_Taken)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 5)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Avg_Wickets_Taken,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
```

**Observation : **
They seem to take 1.4 wickets on an average in the Power Play overs against all teams. This number is highest against Kolkata Knight Riders.
This seems to increase to 2 wickets on an average in the mid overs and it reaches approx. 2.5 in the slog overs.

###**22. Average Wickets Taken - Fielding Second**
```{r}
final_dataset %>%
  filter(first_activity == "bat", Opposition %in% c("CSK","DD","KKR","KXIP","RCB","SRH")) %>%
  select(Opposition, `Wickets Taken - 1st to 6th`, `Wickets Taken - 7th to 15th`, `Wickets Taken - 16th to 20th`) %>%
  group_by(Opposition) %>%
  summarise(`Power Play` = mean(`Wickets Taken - 1st to 6th`)
            ,`Mid Overs` = mean(`Wickets Taken - 7th to 15th`)
            ,`Slog Overs` = mean(`Wickets Taken - 16th to 20th`)) %>%
  gather(Over_Category, Avg_Wickets_Taken, `Power Play`:`Slog Overs`) %>%
  mutate(Over_Category = factor(Over_Category, levels = c("Power Play","Mid Overs","Slog Overs"))) %>%
  ggplot(aes(x = Over_Category, y = Avg_Wickets_Taken)) + geom_bar(stat = 'identity',fill = '#3182bd') + scale_y_continuous(limits = c(0, 5)) + theme(plot.title = element_text(size =10),axis.text.x = element_text(size =7)) +geom_text(aes(label=round(Avg_Wickets_Taken,2)), size = 2.5, position=position_dodge(width=0.2), vjust=-0.25) + facet_wrap(~Opposition)
```

**Observation : **
Mumbai Indians seems to have a very good record in the mid overs where in they have consitently taken about 2.5 wickets on an average.
They seem to struggle early on in taking wickets as the average stands at approx 1.3 wickets.

##**SUMMARY**

1. 2010, 2011, 2013 and 2017 were very good seasons in terms of win percentage. On the contrary, 2009 and 2014 were seasons to forget.
2. They have performed exceptionally well against KKR all throughout the 10 seasons.
3. Their performance has been very good in Bangalore and Kolkata apart from Mumbai which is their home stadium.
4. Winning the toss becomes very crucial against KKR as they seem to win 85% of their matches after winning the toss. On the contrary, its better for them to lose the toss against Delhi as they tend to lose 75% of the matches after winning the toss.
5. They should avoid fielding first in Delhi as they just win 20% of their matches.
6. After winning the toss they should chose to bat first in Chennai and field first in Kolkata as they win all their matches.
7. They should avoid batting first against Sunrisers Hyderabad as they just win 25% of the matches.
