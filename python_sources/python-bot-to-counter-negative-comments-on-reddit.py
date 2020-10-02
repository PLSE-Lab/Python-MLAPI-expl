# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:21:20 2018

@author: Philip Osborne
"""
import praw
from datetime import datetime
import nltk
nltk.download("stopwords")
from textblob import TextBlob
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import re
import os


# Insert your IDs here
reddit = praw.Reddit(user_agent = "",
                username = "",
                password = "",
                client_id = "",
                client_secret = "")

# Setting a specific subreddit
subreddit = reddit.subreddit('FloBotReview')


if not os.path.isfile("posts_replied_to.txt"):
    posts_replied_to = []

else:
    with open("posts_replied_to.txt", "r") as f:
       posts_replied_to = f.read()
       posts_replied_to = posts_replied_to.split("\n")
       posts_replied_to = list(filter(None, posts_replied_to))



    stop_words = set(stopwords.words('english'))

    comment_id = []
    for comment_tracker in subreddit.stream.comments():

        if re.search("FloBot", comment_tracker.body, re.IGNORECASE):
                comment_id = np.append(comment_id,comment_tracker.id)
                comment = comment_tracker.parent()

                if comment.id not in posts_replied_to:
        
        
                    print("Subreddit: ", comment.subreddit)
                    print("Author: ", comment.author)
        
                    print("Text: '", comment.body,"'")
                    print("Score: ", comment.score)
                    print("Sentiment Analysis Subjectivity: ", np.round(TextBlob(comment.body).sentiment.subjectivity,4))
                    print("Sentiment Analysis Polarity: ", np.round(TextBlob(comment.body).sentiment.polarity,4))
        
                    print("---------------------------------\n")
                    subreddit_ref = comment.subreddit
        
                    user = reddit.redditor(str(comment.author))
        
                    comment_history_score = []
                    comment_history_polarity = []
                    comment_history_text_negative = []
                    comment_history_text_normal = []
                    comment_history_subreddit= []
                    comment_history_subreddit_negative = []
                    comment_history_all_negative = []
        
                    for comments in user.comments.new(limit = None):
        
                        comment_history_subreddit = np.append(comment_history_subreddit, comments.subreddit)
                        comment_history_score = np.append(comment_history_score,comments.score)
                        comment_history_polarity = np.append(comment_history_polarity,TextBlob(comments.body).sentiment.polarity)
        
                        words = tokenizer.tokenize(comments.body)
                        for w in words:
                            if w.lower() not in stop_words:
                                if( comments.subreddit == subreddit_ref and TextBlob(comments.body).sentiment.polarity < 0 ):
                                    comment_history_subreddit_negative.append(w.lower())
                                elif( TextBlob(comments.body).sentiment.polarity < 0 ):
                                    comment_history_text_negative.append(w.lower())
                                else:
                                    comment_history_text_normal.append(w.lower())
        
                    comment_history_all_negative = comment_history_text_negative
        
        
                    comment_history = pd.DataFrame( data = {"SubReddit":comment_history_subreddit,"Score": comment_history_score, "Polarity": comment_history_polarity} )
        
                    post_history = []
                    post_history_subreddit = []
                    
                    for posts in user.submissions.new(limit = None):
                        post_history_subreddit = np.append(post_history_subreddit, posts.subreddit)
                        
                    post_history = pd.DataFrame(data={"SubReddit":post_history_subreddit})
        
        
                    print("Beep Boop, ", user, "'s comment has been flagged as negative. FloBot will now analyse their post history.")
                    
                    now = datetime.now()
                    print(user, "'s account is approximately ", np.round((now - datetime.fromtimestamp(user.created_utc)).days/365.25,2)," years old." )
                    
                    print("This user has made ",np.round(len(post_history[post_history['SubReddit']==subreddit_ref])), " original posts to the ",subreddit_ref, "subreddit.")
        
                    negativescore_comment_history = comment_history[comment_history['Score']<0]
                    print("This user has made " , len(comment_history), "comments in total. The percentage of these that have a negative score is:")
                    print( np.round((len(negativescore_comment_history)/ len(comment_history))*100,2),"%")
        
        
                    negativepolarity_comment_history = comment_history[comment_history['Polarity']<0]
                    print("The percentage of comments made by this user that contain words considered negative is:")
                    print( np.round((len(negativepolarity_comment_history)/ len(comment_history))*100,2),"%")
        
        
                            
        
                    negativepolarity_subreddit_comment_history = comment_history[ (comment_history['Polarity']<0) & (comment_history['SubReddit'] == subreddit_ref)]
                    subreddit_comment_history = comment_history[comment_history['SubReddit'] == subreddit_ref]
                    print("This user has made " , len(subreddit_comment_history), "comments to this subreddit. The percentage of these that contain words considered negative is:")
                    print( np.round((len(negativepolarity_subreddit_comment_history)/ len(subreddit_comment_history))*100,2),"%")
        
        
                    pos_res_negative = nltk.FreqDist(comment_history_all_negative)
                    pos_res_negative_top = pos_res_negative.most_common(10)
                    print("This user used the following words most frequently in their negative comments: ")
                    print([x[0] for x in pos_res_negative_top])
        
                    posts_replied_to = np.append(posts_replied_to,comment_tracker.id)
        
                    print("-.-.-.-.-.-.-.-.-.-.-")
        
        with open("posts_replied_to.txt", "w") as f:
            for comment_ref in posts_replied_to:
                f.write(comment_ref + "\n")
                                                