#!/usr/bin/env python
# coding: utf-8

# <img src="https://techcrunch.com/wp-content/uploads/2015/08/googleanalytics.jpg" alt="corona" width="700">
# 
# <br>
# I thought it might be helpful to share some thoughts on best practices for challenges involving general analysis. My opinions come from experience being both the person reading reports and the person making reports - and the person told to improve and resubmit his report. Of course this notebook represents a single point of view, and you are probably best served by gathering many perspectives of what "good" looks like. 
# 
# <br>
# 
# A note specific to the CTDS ahllenge: We are fortunate to have accomplished Kagglers as host and judges for this challenge. I encourage you to research their work and take your cues from them as well. Good luck!
# 

# # 1. Lead with the insights.
# 
# In case you read no further, I offer this idea as the one thing to make your report more impactful:
# 
# > #### Lead with the best insights from your analysis. Put the bottom line up front.
# 
# This point is important, and not natural for scientists. We like to take the audience through our line of reasoning, with open minds, supported by data and charts with a bit of explanation. After they've followed along we present our conclusions.
# 
# Sometimes that's useful, but not so much here! Your audience consists of people looking for vital information in a limited amount of time. They'll be reading through a stack of reports looking for something that seems useful. So it's best to start with your conclusion, get the reader's interest, and give them a frame of reference from which to "hang" supporting information. It's acutally easier for someone to digest information when they know the overall storyline. Once you've set it up, you can then explain the details and show them how you got there. 
# 
# Here's the beginning of my report from the 2019 NFL Punt Analytics challenge:
# 
# <hr>
# 
# > ### Summary
# 
# > This report represents my analysis for the NFL Punt Analytics Competition. It is my opinion that based on the data provided, changing two rules for punt plays could result in up to 8 fewer concussions per year. The two proposed changes are as follows:
# 
# >  - <i>Move the ball forward 10 yards after a Fair Catch.</i> After fair catch of a scrimmage kick, play would start 10 yards forward from the spot of catch, or the succeeding spot after eforcement of penalties. This would apply when the receiving team elects to put the ball in play by a snap.
# 
# >  - <i>Penalize blindside blocks with forcible contact to any part of a defenseless player's body.</i> Defenseless players as described by NFL rules include players receiving a blindside block from a blocker whose path is toward or parallel his own end line. Prohibited contact for punt plays would include blindside blocks with forcible contact to any part of a defenseless player's body, including below the neck.
# 
# > The figure below shows the potential reduction in concussions based on 2016-2017 data and associated assumptions.
# 
# 
# <img src="https://s3.amazonaws.com/nonwebstorage/headstrong/redux.png" alt="chart" height="400" width="600">
# 

# You can see there's minimal setup and then we get right to the point. For this challenge, you might consider starting right away with your findings, based on data of course.
# 

# # 2.  Find a question that's worth answering.
# 
# Sometimes it can be hard to know where to focus. Overall I recommend you find issues that will provide useful information to the data science community. Some ideas off the top of my head:
# 
#  - Which interviews had the most engagement and what did they have in common?
#  - What trends are visible over the past year and what might they say about the coming year?
#  - What advice/information regarding Kaggle should be shared with the broader data science community? 
# 

# # 3. Spend time on your visualizations. 
# 
# The art and science of visualization has become a field of study on it's own with many, many things to learn. For now, here are some things you can consider to help your visualizations reinforce your message.
# 
# - Keep it simple. Your audience should understand what they're looking at within a few seconds.
# - Use color to reinforce your message. Color is one of the "preattentive attributes" that people pick up on right away. A bar chart with mostly gray bars and one bar highlighted in color is a great way to focus attention. Along those lines, try not to use default seaborn bar charts with the rainbow of colors. Too much color can be worse than no color.
# - Tell the reader what to takeawy from the chart. That might not sound right at first, but your chart should reinforce your data-driven conclusions. It's ok to put that point front and center on the page. The audience can ask questions and make other points if they see things differently.
# 
# Here'an example from Tom Bresee's [Next-Gen EDA](https://www.kaggle.com/tombresee/next-gen-eda) notebok for the recent NFL Databowl competition. It uses a super-clean layout with great use of color. I find it much more effective than a regular old bar chart. In  my opinion it is one of the best examples on Kaggle of a minimal, effective visualization.
# 
# 
# <img src="https://i.imgur.com/cCgC2NM.png" alt="schedule" width="600" height="500">
# 
# <br>
# I'm also a big fan of interactive charts. Plotly and Bokeh are good choices. Using a wrapper like Holoviews or hvplot makes interactive charts even easier.
# 

# # 4. Make your notebook easy to read. 
# 
# Notice how I numbered my major headers for the notebook? That's one way to help keep readers keep their place as they scroll through. Kaggle has a great feature now that makes a table of contents off to the side of the notebook based on your headers. Even better!
# 
# 
# Here are some other things you can try.
# 
#  - Use html tags and hierarchy to organize your notebook. Markdown is great for convenience. It's also flexible by allowing direct use of HTML tags. Here's an example of code for custom HTML to make section headers and structure that stand out. All you need is code at the top of your notebook and then matching tags in the markdown cells.
# 

# In[ ]:


# %%HTML

# This is the code for a code cell that sets the formats. Put it at the top of your notebook.

# <style type="text/css">

# div.h2 {
#     background-color: steelblue; 
#     color: white; 
#     padding: 8px; 
#     padding-right: 300px; 
#     font-size: 24px; 
#     max-width: 1500px; 
#     margin-top: 50px;
#     margin-bottom:4px;
# }



# This is the actual HTML code you would put in a markdown cell. I can't do it here because
# it cuts the rest of the notebook off for some reason.

# <div class=h2>1. Section Title. </div>
# Blah blah blah.


#  - Make your code easy to read. One of the best resources I've found is this Kaggle notebook [Six steps to more professional data science code](https://www.kaggle.com/rtatman/six-steps-to-more-professional-data-science-code) by Rachael Tatman. The sections on "readable" and "stylish" have great info.

#  - Another thing you might consider is to hide your code. Kaggle makes it easy with the "hide input" option for each cell. Normally, code is what we're all about. Part of what makes Kaggle notebooks great is that we learn a ton from each other's code, and the last thing you probably want to do is hide it. But analytics reports are different. It's more important for your readers to first get the overall story if they're so inclined. Once you pass that first lookover, you can expect someone will dig into your code.

# # 5. Don't forget the stated criteria.
# 
# Scores for analytics challenges are subjective by their very nature. Even so, you will need to meet the requirements stated in the competition. For example, if the host lists "storytelling" as a criterion, do your best to have cohesive flow throughout your notebook.

# # 6. Resources and Closing.
# 
# Here are some good report-style notebooks from past analytics challenges, IMO:
# 
#  - https://www.kaggle.com/philippsinger/nfl-playing-surface-analytics-the-zoo
#  - https://www.kaggle.com/gaborfodor/summary-budapest-pythons
#  - https://www.kaggle.com/jpmiller/nfl-punt-analytics/#data
#  - https://www.kaggle.com/erikbruin/recommendations-to-passnyc-1st-place-solution
#  
#  
# Favorite sources of good visualization:
# 
# - https://fivethirtyeight.com/
# - https://www.nytimes.com/section/upshot
# - https://flowingdata.com/
# - http://www.storytellingwithdata.com/
# - https://www.edwardtufte.com/tufte/
# 
# 
# I hope these ideas are helpful and add to the quality of everyone's reports. I also hope to continue learning from this great community!
