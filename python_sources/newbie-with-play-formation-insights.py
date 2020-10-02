#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Hello! I'm new to this so be patient.  I don't even know how to creat a kernal :(  I'm an MBA who doesn't know how to code (at all). I took this side project as a way to improve my analytics skills over break.  I've found a couple things that I think are useful, but have done my work in Excel and RapidMiner.  I'm trying to learn R and Python but it looks like it will be a long journey.
# 
# **Getting Started**
# I spent about a day getting familiar with each of the csv files, the attributes, and the relationships within the data.  I started creating a number of new attributes in files such as Fair Catch, Downed, Out of Bounds, or Touchback.  I also looked into whether or not the punt was from the punting team's side of the field.  
# 
# This gave me some basic info:
# <img src=https://imgur.com/5P5TLIN.png>
# 
# Own Side of Field	
# <img src=https://imgur.com/4YB64xW.png>
# 
# I also found that night games had nearly double the concussion rate
# <img src=https://imgur.com/dK5SR50.png>
# 
# **Re-focusing on the problem**
# Ultimately I realized while a lot of this is helpful, I'm not sure much of it is useful in adapting the game.  In addition, if I want to predict concussions on punt plays, a lot of this data like fair catch is data leakage.  
# 
# Ultimately, I decided to explore play formation and the impact on concussions.  To do so, I needed to get my hands around the players and roles on the field for various plays.
# 
# **Manipulating the Play_Player_Role_Data file**
# So many player roles!  I wanted to simplify the player roles.  Ultimately I made several replacements.  For each of the positions, I got rid of the numbers associated with the role.  For instance, PDL3 became PDL.  I then looked into concussion rates by player.  
# 
# The punting team incurs most concussions and concussion rates vary by role, but interior linemen have a lot.  
# <img src=https://imgur.com/DGtKtKG.png>
# 
# For the return team, most concussions were on the punt returner.  
# <img src=https://imgur.com/bmffj7o.png>
# 
# 
# **Evaluating Plays by Formation**
# Since I don't know how to code, I started creating columns in the play_information file.  I used countif statements to identify how many players at each position were involved in each play.  Now the fun comes in!  
# 
# My hypothesis was that players running at fast speeds down the field are going to get concussions.  And how do these interior linemen that have so many concussions get at high speeds? They must not be blocked, right?
# 
# So I created a couple of different attributes to add to my play data.  One attribute was Number of Defenders in the Box.  This is defined as a count of PDL, PDR, PLL, PLR, and PLM players.  The other attribute is the Side Overload.  This is defined as the absolute value between players on the left side in the box and players on the right side in the box.  | PDL + PLL - PDR - PDR | 
# 
# A simple pivot table showed some promising signs.  There was positive correlation between Side Overload and Concussions.  There was negative correlation between Number of Defenders in the Box and Concussions.
# 
# Overloaded Def	Concussion	Plays	%
# <img src=https://imgur.com/mt9VNot.png>
# 
# DEF in the Box	Concussions	Plays	%
# <img src=https://imgur.com/7aNSMhE.png>
# 
# Illustration of the box
# <img src=https://imgur.com/7eXkY9N.png>
# 
# Relationships between concussions and formation
# <img src=https://imgur.com/tUGG9OZ.png>	
# 
# <img src=https://imgur.com/iL4Nw4q.png>
# 
# **Modeling Play Formation**
# I used RapidMiner to then create a logistic regression using play formation.  I filtered out outliers like <6 players in the box or >9.  I created the model using 70% training data and tested it on the other 30%.
# 
# ****Of note, I filtered out several plays that I considered outliers.  This included Overload >=4, DEF BOX <6, DEF BOX>8.  It is normal for the defense to to have 6-8 in the box, with either doubling up on the gunners or putting single coverage on the gunners.  I also used a filter for "no punt" because of penalty or other instance (fake punt / blocked punt) and not using out of bounds kicks or touchbacks.  I calculated "no punt" based on punt yardage.  I had calculated punt yardage as a field for every play based on the PlayDescription field where it will say "punts XX yards" when there is a punt.  This represented 220 observations.  The other filters excluded 150 observations.
# 
# <img src=https://imgur.com/1HH9AEt.png>
# 
# The successfully predicted 3/11 concussion plays.  Given how rare of an event a concussion is on any given play, I thought this was pretty good.  
# <img src=https://imgur.com/hITvquM.png>
# 
# Therefore my recommendation will be focused on making the formation include 8 players in the box and not overloading one side of the box.
# 
