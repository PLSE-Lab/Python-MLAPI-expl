#!/usr/bin/env python
# coding: utf-8

# Please refer to the [problem statement](https://storage.googleapis.com/coding-competitions.appspot.com/HC/2019/hashcode2019_qualification_task.pdf) for the full elaboration of the problem.
# 
# To summarise the problem statement:
# - You are given a list photos.
# - Each photo is either "horizontal" or "vertical".
# - Each photo has a list of tags.
# - If the photo is vertical, you will need to merge it with another vertical photo to form a combined photo. The combined photo will contain all the tags from both photos.
# - You will need to provide a sequence of combined and horizontal photos to that maximises the score of your solution.
# - The score of your solution is the sum of scores between all pairs of neighbours in your produced sequence.
# - The score between neighbouring combined/horizontal photos is the minimum of
# 
#   - tags found in both neighbours
#   - tags found only in the first neighbour
#   - tags found only in the second neighbour
# 
# In this notebook, I present a baseline solution that only takes in information of the tag length and not the tags, along with some data analysis and visualisations. This should be sufficient for you to write an algorithm that is improved from this baseline.

# In[ ]:


get_ipython().run_line_magic('reset', '-sf')
import random
import collections
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

random.seed(42)

get_ipython().system('ls /kaggle/input/hashcode-photo-slideshow/')


# # Parse input
# We first need to parse the input from the text file.

# In[ ]:


filepath = "/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt"
with open(filepath) as f:
    pictures = [row.strip().split() for row in f.readlines()][1:]


# In[ ]:


get_ipython().system('head -3 "/kaggle/input/hashcode-photo-slideshow/d_pet_pictures.txt"')


# Input format
# ```
# number_of_photos
# horizontal_or_vertical number_of_tags tag1 tag2 tag3 ...
# horizontal_or_vertical number_of_tags tag1 tag2 tag3 ...
# ...
# ```

# In[ ]:


pic_tags = {}  # maps idx to tags
horizontal_photos = []  # horizontal photos only
vertical_photos = []  # vertical photos only
for i,picture in enumerate(pictures):
    pic_tags[i] = set(picture[2:])
    if picture[0] == "H":
        horizontal_photos.append(i)
    elif picture[0] == "V":
        vertical_photos.append(i)
print(len(vertical_photos), len(horizontal_photos))


# There are 60000 vertical photos and 30000 horizontal photos.

# # Score calculation
# We define a set of functions that we can repeatedly use to calculate the score. 
# 
# While you can submit to Kaggle for the computation of the score, writing these functions helps to increase the rate at which you iterate your solution. It is also recommended one of the team members writes such a function during the actual qualifiers which can be shared among the team of up till four.

# In[ ]:


def calc_tags_pair_score(tags1, tags2):
    # given two sets of tags, calculate the score
    return min(len(tags1 & tags2), len(tags1 - tags2), len(tags2 - tags1))

def calc_idxs_pair_score(idxs1, idxs2):
    # given two tuples of indices, calculate the score
    return calc_tags_pair_score(
        set.union(*[pic_tags[idx] for idx in idxs1]),
        set.union(*[pic_tags[idx] for idx in idxs2]))

def calc_idxs_pair_score_max(idxs1, idxs2):
    # given two tuples of indices, calculate the maximum possible score by tag length
    return min(len(set.union(*[pic_tags[idx] for idx in idxs1])),
               len(set.union(*[pic_tags[idx] for idx in idxs2])))//2

def calc_sequence(idxs_lst):
    # given the sequence of indices, calculate the score
    check_validity(idxs_lst)
    score = 0
    for before, after in zip(idxs_lst[:-1], idxs_lst[1:]):
        score += calc_idxs_pair_score(before, after)            
    return score

def calc_sequence_max(idxs_lst):
    # given the sequence of indices, calculate the score
    check_validity(idxs_lst)
    score = 0
    for before, after in zip(idxs_lst[:-1], idxs_lst[1:]):
        score += calc_idxs_pair_score_max(before, after)            
    return score

def check_validity(idxs_lst):
    all_pics = [idx for idxs in idxs_lst for idx in idxs]
    if len(all_pics) != len(set(all_pics)):
        print("Duplicates found")
    all_verts = [idx for idxs in idxs_lst for idx in idxs if len(idxs) == 2]
    if (set(all_verts) - set(vertical_photos)):
        print("Horizontal photos found in vertical combinations")
    all_horis = [idx for idxs in idxs_lst for idx in idxs if len(idxs) == 1]
    if (set(all_horis) - set(horizontal_photos)):
        print("Vertical photos found in horizontal arrangement")


# `calc_sequence` computes the score like as per required by the problem statement.
# 
# I would also like to introduce `calc_sequence_max`. This is the theoretical maximum if we can control the tags for a given arrangement of combined/vertical photos. This will be elaborated [later](https://www.kaggle.com/huikang/hc-2019q-eda-and-baseline-soln#Theoretical-maximum).

# In[ ]:


idxs_list = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])] 
idxs_list.extend([(a,) for a in horizontal_photos])
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# # Match by tag length
# If we match the vertical photos by tag length, this is the performance that we can get. This step does not take into account of what tags do the photos have.

# In[ ]:


random.shuffle(vertical_photos)
vertical_photos.sort(key=lambda idx: len(pic_tags[idx]))
idxs_list_combined = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])]
idxs_list = idxs_list_combined + [(a,) for a in horizontal_photos]
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# # Arrange by tag length
# If we arrange the combined/horizontal photos by tag length, this is the performance that we can get. This step also does not take into account of what tags the photos have. 

# In[ ]:


random.shuffle(idxs_list)
idxs_list.sort(key = lambda idxs: sum([len(pic_tags[idx])//2 for idx in idxs]))
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# Without looking at the tag, and merely using the information of the tag length, we have obtained a score of 215408.
# 
# We also see that the maximum possible score of the arrangement also changes with arrangement.

# # Exploratory Data Analysis
# In this section, we explore the given data. This is only dependent on the input and does not depend on how we solve the problem.

# In[ ]:


tags_set = sorted(set(tag for idx,tags in pic_tags.items() for tag in tags))
tags_counter_all = collections.OrderedDict((tag,0) for tag in tags_set)
tags_counter_horizontal = collections.OrderedDict((tag,0) for tag in tags_set)
tags_counter_vertical = collections.OrderedDict((tag,0) for tag in tags_set)
for idx in horizontal_photos:
    for tag in pic_tags[idx]:
        tags_counter_horizontal[tag] += 1
        tags_counter_all[tag] += 1
for idx in vertical_photos:
    for tag in pic_tags[idx]:
        tags_counter_vertical[tag] += 1
        tags_counter_all[tag] += 1


# We wonder if there are tags that appear more frequently in vertical photos than in horizontal photos. This is not the case, as we see the tags appearing at an almost equal frequency between vertical and horizontal photos.

# In[ ]:


plt.figure(figsize=(14,4))
plt.scatter([v for k,v in tags_counter_vertical.items()],
            [v for k,v in tags_counter_horizontal.items()], label="one tag")
plt.xlabel("freqency of tags in vertical photos")
plt.ylabel("freqency of tags in horizontal photos")
plt.title("total number of tags: " + str(len(tags_counter_all)))
plt.legend()
plt.show()


# As we cannot see the number of dots in the scatter plot, we plot a histogram instead. As there are two times more vertical photos compared to horizontal photos, we halve the frequency of the vertical photos for presentation.

# In[ ]:


plt.figure(figsize=(14,4))
bins = np.arange(0,6000,50)
plt.hist([[count for tag,count in tags_counter_horizontal.items()],
          [count/2 for tag,count in tags_counter_vertical.items()]],
         bins=bins, stacked=True,
         label=["number of tags on horizontal photos in the frequency bucket",
                "number of tags on vertical photos in the frequency bucket"])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# In[ ]:


tags_freq_mean = np.mean(list(tags_counter_all.values()))
more_frequent_tags = set([tag for tag,count in tags_counter_all.items() 
                          if count > tags_freq_mean])
less_frequent_tags = set([tag for tag,count in tags_counter_all.items() 
                          if count <= tags_freq_mean])
int(tags_freq_mean), len(more_frequent_tags), len(less_frequent_tags)


# We wonder if each vertical photo has an equal proportion of the more common tags and less common tags. This is not the case, as we see a spread of such proportion.

# In[ ]:


plt.figure(figsize=(14,4))
plt.scatter([len([tag for tag in pic_tags[idx] if tag in more_frequent_tags]) 
             + np.random.uniform() for idx in vertical_photos],
            [len([tag for tag in pic_tags[idx] if tag in less_frequent_tags]) 
             + np.random.uniform() for idx in vertical_photos], 
            s = 1, alpha=0.2, label = "one vertical photo")
plt.xlabel("number of more common tags")
plt.ylabel("number of less common tags")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# We are also interested in the distribution of the number of tags in each picture

# In[ ]:


plt.figure(figsize=(14,4))
plt.hist([len(pic_tags[idx]) for idx in horizontal_photos], bins=range(20), alpha=0.5,
         label="distribution of number of tags of horizontal photos")
plt.hist([len(pic_tags[idx]) for idx in vertical_photos], bins=range(20), alpha=0.2,
         label="distribution of number of tags of vertical photos")
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                       xytext=(0, 0), textcoords='offset points', 
                       ha='center', va='bottom', fontsize=8)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# # Solution analysis
# We also conduct a series of analysis on our solution to see what can be improved.

# We want to know how distribution of the score varies along the sequence. As we have sorted the photos in an increasing manner, the mean of the distribution of the obtained score increases along with the sequence length. However, we are still far from obtaining the optimum score of the sequence.

# In[ ]:


plt.figure(figsize=(14,4))
plt.plot([calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[:-1], idxs_list[1:])], alpha=0.5, 
         label="score of neighbours in the sequence")
plt.plot([calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[:-1], idxs_list[1:])], alpha=0.5, 
         label="maximum possible score of neighbours in the sequence")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# In the EDA we have seen the distribution of the number of tags in each vertical photo. We want to know the distribution of the number of tags in each combined photo.

# In[ ]:


plt.figure(figsize=(14,4))
plt.hist([len(pic_tags[idxs[0]] | pic_tags[idxs[1]])
          for idxs in zip(vertical_photos[::2], vertical_photos[1::2])], 
         bins=range(36), alpha=0.5,
         label="distribution of number of tags of combined photos")
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                       xytext=(0, 0), textcoords='offset points', 
                       ha='center', va='bottom', fontsize=7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# We also want to known within in combined pair of vertical photos, how many tags overlap with each other. This shows that there is room for improvement if we can merge vertical photos such that there is minimal overlap in the tags.

# In[ ]:


plt.figure(figsize=(14,4))
plt.hist([len(pic_tags[idx1] & pic_tags[idx2]) for idx1,idx2 in 
          zip(vertical_photos[::2], vertical_photos[1::2])], bins=range(20), alpha=0.5,
         label="distribution of number of overlapping tags " + 
               "between neighbours of vertical photos sorted by number of tags")
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                       xytext=(0, 0), textcoords='offset points', 
                       ha='center', va='bottom', fontsize=8)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# The following shows the distribution of scores between each possible pair of neighbour
# - (combined) vertical to (combined) vertical
# - (combined) vertical to horizontal (or vice-versa)
# - horizontal to horizontal
# 
# As the the combined vertical photos have more tags, we see that highest scoring neighbours are vertical-vertical transitions.

# In[ ]:


plt.figure(figsize=(14,4))
plt.hist([[calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 4],
          [calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 3],
          [calc_idxs_pair_score(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 2]],
         bins=range(20), alpha=0.5,
         label=[
        "distribution of scores for vertical-vertical neighbours",
        "distribution of scores for horizontal-vertical neighbours",
        "distribution of scores for horizontal-horizontal neighbours"], 
         stacked=True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# This is the maximum possible score of the transition. In the optimal solution, the graph below should look very similar to the graph above.

# In[ ]:


plt.figure(figsize=(14,4))
plt.hist([[calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 4],
          [calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 3],
          [calc_idxs_pair_score_max(idxs1,idxs2) for idxs1,idxs2 in 
          zip(idxs_list[::2], idxs_list[1::2]) if len(idxs1) + len(idxs2) == 2]],
         bins=range(20), alpha=0.5,
         label=[
        "maximum possible scores of the transition between vertical-vertical pair",
        "maximum possible scores of the transition between vertical-horizontal pair",
        "maximum possible scores of the transition between horizontal-horizontal pair"], 
         stacked=True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


# # Theoretical maximum
# Understanding the theoretical maximum make us better understand how after away are we from the optimum score. To calculate the theoretical maximum
# - We assume that we can control all the tags as we wish
# - We are still constrained by the number of tags of each photo
# 
# The ideal distribution of tags happens when all three of the following are equal
# - tags found in both neighhours
# - tags found only in the first neighhour
# - tags found only in the second neighhour
# 
# Therefore, the maxmium possible score between the neighbours is the half of the number of tags (and rounded down) of the neighbour with a smaller number of tags.
# 
# The upper bound of the theoretical maximum is therefore **443406**. The actual theoretical maximum is slightly lower because we need to transition between combined/horizontal photos of different tag length, and also account for the first and last photo which has only one neighbour.

# In[ ]:


(sum(len(pic_tags[idx]) for idx in vertical_photos)//2 +  sum(len(pic_tags[idx])//2 for idx in horizontal_photos))


# # Write submission
# We now write our submission according to the prescribed format. The photos start with an index of zero.

# In[ ]:


submission_lines = []
submission_lines.append(str(len(idxs_list)))
for idxs in idxs_list:
    submission_lines.append(" ".join([str(idx) for idx in idxs]))


# In[ ]:


with open("submission.txt", "w") as f:
    f.writelines("\n".join(submission_lines))


# In[ ]:


get_ipython().system('head -3 submission.txt')


# Submission format
# ```
# nrows
# photo_id_1v photo_id_2v
# photo_id_h
# ...
# ```

# We reprint our score here for quicker reference in the future.

# In[ ]:


calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)

