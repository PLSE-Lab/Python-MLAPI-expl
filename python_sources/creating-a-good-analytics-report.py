#!/usr/bin/env python
# coding: utf-8

# It's great to see Kagglers contributing this way to fight the outbreak. I thought it might be helpful to share some thoughts on what makes a good report for this type of thing. My opinions come from experience being both the person reading reports and the person making reports. Of course this notebook represents a single point of view, and it's best to gather many perspectives of what "good" looks like.
# 
# NOTE: I originally wrote this notebook for the NCAA challenge. I've reworked it here to support this effort. My intent is to focus more on making a useful, high-impact report vs. winning a competition, though ideally the two wil be highly correlated. 
# 

# # 1. Lead with the insights.
# 
# In case you read no further, I offer this idea as the one thing to make your report more impactful:
# 
# > #### Put your bottom line up front. Lead with the best insight from your analysis.
# 
# This point is important, and not natural for scientists. We like to take the audience through our line of reasoning, with open minds, showing them a series of charts with some explanation. After they've followed along we present our conclusions.
# 
# Sometimes that's useful, but not so much here! Your audience consists of people looking for vital information in a limited amount of time. They'll be reading through a stack of reports looking for something that seems useful. So start with your conclusion, get the reader's interest, and give them a frame of reference from which to "hang" supporting information. Then, explain the details and show them how you got there. 
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

# Minimal setup and right to the heart of it. In the case at hand you might consider starting with your take on the existing research, based on data of course.

# # 2.  Find a question that's worth answering.
# 
# Sometimes it can be hard to know where to focus. That's not so much the case here with the sponsors giving us an extensive list and subtopics to explore. One possiblity is that within an area of interest, find where the research intersects with what everyone is talking about. Some ideas off the top of my head, relevant to the US as that's where my experience is based:
# 
#  - How has media's influence on public perception changed relative to the days of newspapers and broadcast TV?
#  - What will be the long term effects of the outbreak on the wealth gap in developed countries?
#  - How does a nation's health care structure influence the short and long term effects of disease outbreaks?
#  
#  Another avenue may be to research two or more distinct areas and show how they complement one another.

# # 3. Spend time on your visualizations. 
# 
# The art and science of visualization has become a field of study on it's own with many many things to learn. For now, here are some things you can do to make sure your visualizations reinforce your message.
# 
# - Keep it simple. Your audience should understand what they're looking at within a few seconds.
# - Use color to reinforce your message. Color is one of the "preattentive attributes" that people pick up on right away. A bar chart with mostly gray bars and one bar highlighted in color is a great way to focus attention. Along those lines, please don't use default seaborn bar charts with the rainbow of colors. Too much color can be worse than no color.
# - Tell the reader what to think. That might not sound right at first, but your chart should reinforce a point and it's ok to put that point front and center on the page.
# 
# Here'an example from Tom Bresee's [Next-Gen EDA](https://www.kaggle.com/tombresee/next-gen-eda) notebok for the recent NFL Databowl competition. It uses a super-clean layout with great use of color. I find it much more effective than a regular old bar chart.
# 
# <img src="https://i.imgur.com/cCgC2NM.png" alt="schedule" width="600" height="500">
# 

# # 4. Make your notebook easy to read. 
# 
# Notice how I numbered my major headers for the notebook? That's one way to help keep your place as you scroll through. Here are some other things you can try.
# 
#  - You can use html tags and hierarchy to organize your notebook. Markdown is great for convenience and it's also flexible if you apply HTML tags. Here's an example of code for custom HTML to make section headers and structure that stand out. 
# 

# In[ ]:


# This is the code for a code cell that sets the formats.

# %%HTML
# <style type="text/css">

# div.h1 {
#     font-size: 32px; 
#     margin-bottom:2px;
# }
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


# This is the actual HTML for a formatted markdown cell.

# <div class=h2>1. Section Title. </div>
# <div class=h4>Major headers. </div>
# Blah blah blah.


#  - Make your code easy to read. There's an explicit request to do that for this challenge. One of the best resources I've found is this notebook [Six steps to more professional data science code](https://www.kaggle.com/rtatman/six-steps-to-more-professional-data-science-code). The sections on "readable" and "stylish" have great info.

#  - Another thing you might consider is to hide your code. Kaggle makes it easy with the "hide input" option. Normally, code is what we're all about and we're asked to have readable code. The last thing you probably want to do is hide it. But some of your readers don't code and even if they do, it's more important for them to first get the overall story if they're so inclined. Once you pass that first lookover, you can expect someone will dig into your code. 

# # 5. Don't forget the stated criteria.
# 
# My perception is that the grading criteria were not strictly followed in previous analytics challenges. It's really more about what ideas can be used by the host to further their goals. You will need to meet the minimum bar, however, to be considered for selection. For example, be sure to have a section on pros and cons of your approach.
# 
# 

# # 6. Closing.
# I hope these ideas are helpful and add to the quality of everyone's reports!