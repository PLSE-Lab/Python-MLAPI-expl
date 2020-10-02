#!/usr/bin/env python
# coding: utf-8

# # Editorial
# 
# Please refer to my [Exploratory Data Analysis](https://www.kaggle.com/huikang/hc-2019q-eda-and-baseline-soln) for my summary of the problem and some insights on the dataset.
# 
# For any approach, the steps can be classified into two stages, both of which should be improved
# - You need to **match** vertical photos into "combined" photos.
# - You need to **arrange** the combined/horizontal photos for maximum score.

# On how the vertical photos are [**matched**](#ID_MATCHED)
# - Conducting a data analysis on the vertical photos, we see that there are signficant overlap in the number of tags between vertical photo pairs.
# - I needed to match the photos in the way that it minimise the amount of overlapping tags.
# - I avoid pairings that result in odd number of tags. This improved the solution my almost a thousand from 440k. This improved the result slightly, but adds 5 minutes to the computation time.
# - I try to pair pictures so that their tags combined is close to 20 (and even), the average. The hypothesis is that with a similar amount of tags, it is easier to find the next neighbour with the optimum score. This also require the increased computation time.
# 
# On how the photos are [**arranged**](#ID_ARRANGED)
# - This is now a travelling salesman problem (TSP). Each node is one combined/horizontal photo. A score is assigned between each combination of combined/horizontal pairs, like how a distance is assigned between each location pairs in the traditional TSP. Instead of minimising distance in the traditional TSP, we maximise the score.
# - A greedy approach is used.
# - The greedy search starts from the combined/horizontal photos with the most number of tags. We do this until no more combined/horizontal photos are left.
# - We do not search in all the remaining photos. Instead of searching from all the remaining photo, we search among the remaining K=10000 photos.
# - We can also stop when we have obtained the maximum possible score, calculated from the number of tags of the preceding photo. This drastically sped up our computation time from 1 hour with K=10000 to 6 minutes.

# Performance
# - With a running time of 6 minutes from scratch, I have obtained a score of **441088**.
# - The [upper bound of the theoretical maximum](https://www.kaggle.com/huikang/hc-2019q-eda-and-baseline-soln#Theoretical-maximum) is 443406.
# 
# Competition considerations
# - The entire duration of Hash Code qualifiers is 4 hours, including reading the question and submission. A group of up to four people work on the problem.
# - We can reduce K further down to 100. This reduces the computation time to 133 seconds, scoring 412858.
# - The [22nd place](https://medium.com/@danieleratti/how-we-placed-1st-in-italy-and-22nd-in-the-world-google-hashcode-2019-e59e52232b4e)  of the actual qualifiers scored 347793. Some grandmasters on [Codeforces](https://codeforces.com/blog/entry/65617) do manage to hit 440k for this problem during the qualifiers. [Errichto](https://codeforces.com/blog/entry/65617?#comment-496761) managed to run the algorithm in 2 minutes during the competition, using C++ bitsets.
# - Realistically, I do not expect teams to come up with the perfect algorithm. The most important part is implementing the greedy search effectively. Algorithms that attempt to conduct the greedy search on all the remaining combined/horizontal photos will run out of time (estimated 6 hours). It is important to understand that for each node there is a maximum score, and you can stop finding stop once you obtained the score (or close to it).
# - The competition is done in a team. You want to perform better than four people coding individually. There is also a trade-off in collaborating because communication takes up time and effort. This is something your team need to discuss, plan, and practice.

# Further improvements
# - As mentioned, we are already very close to the theoretical maximum of 443406.
# - You can try local optimisation on the sequences (as presented in other notebooks). To make this optimisation an effective use of time, you can ignore neighbours that already have an optimum score.
# - You might want to change the way how vertical photos are merged. Currently, vertical photos are usually merged with vertical photos with the same number of tags, or merged such that the combined number of tags is close to 20. You might want to change this distribution.
# - You can consider the frequency of the tags when you arranged during merging. You might want all pairs of vertical photos to share a similar distribution of common tags and less common tags.
# - You can consider the frequency of the tags when you arranged during arranging. For example, in the consideration of the next neighbour, you might want to prefer the intersection tags between adjacent photos (matched-vertical or horizontal) to use the tags that appear less frequently. This way, the remaining neighbours tend to have more common tags, making it easier to find optimium neighbours among themselves.
# - You can also try this algorithm on [other datasets](https://storage.googleapis.com/coding-competitions.appspot.com/HC/2019/qualification_round_2019.in.zip) in the competition.

# In[ ]:


get_ipython().run_line_magic('reset', '-sf')
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

random.seed(42)
get_ipython().system('ls /kaggle/input/hashcode-photo-slideshow/')


# In[ ]:


# paramters for maximum performance (10 minutes)
REARRANGE_FOR_MERGE = True
MERGE_WINDOW = 10000
ARRANGE_WINDOW = 10000

# # paramters for quick compute (2 minutes)
# REARRANGE_FOR_MERGE = False
# MERGE_WINDOW = 100
# ARRANGE_WINDOW = 100


# # Parse input

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


# # Score calculation

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


# In[ ]:


idxs_list = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])] 
idxs_list.extend([(a,) for a in horizontal_photos])
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# In[ ]:


random.shuffle(idxs_list)
idxs_list.sort(key = lambda idxs: sum([len(pic_tags[idx]) for idx in idxs]))
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# # MATCHING vertical photos <a id='ID_MATCHED'></a>

# In[ ]:


# match vertical photos by tag length
random.shuffle(vertical_photos)
vertical_photos.sort(key=lambda idx: len(pic_tags[idx]))
idxs_list = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])]
idxs_list.extend([(a,) for a in horizontal_photos])
idxs_list.sort(key = lambda idxs: sum([len(pic_tags[idx]) for idx in idxs]))
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# In[ ]:


vertical_tmp = vertical_photos[::-1]  # start from photo with most tags
if REARRANGE_FOR_MERGE:  
    # so we can easily match photos with more tags with photos with less tags
    vertical_photos[0::2] = vertical_tmp[:30000]
    vertical_photos[1::2] = vertical_tmp[30000:][::-1]
vertical_tmp = vertical_photos
vertical_photos = [vertical_tmp[0]]
vertical_tmp = vertical_tmp[1:]

for i in tqdm(range(len(vertical_tmp))):
    idxs1 = vertical_photos[-1]
    best = -9999
    best_next_ptr = 0
    cnt = 0
    for j,idxs2 in enumerate(vertical_tmp):
        if len(vertical_photos)%2 == 0:  # we do not need to consider between pairs
            break
        if best == 0:
            # we have found an optimal match
            break
        if cnt > MERGE_WINDOW:
            # early stopping in the search for a paired photo
            break
        score = -len(pic_tags[idxs1] & pic_tags[idxs2])
        num_tags_if_paired = len(pic_tags[idxs1] | pic_tags[idxs2])
        if num_tags_if_paired%2 == 1:  
            # penalise if the total number of tags is odd
            score = min(score,-0.9)
        if num_tags_if_paired > 22 and REARRANGE_FOR_MERGE:  
            # to encourage the total number of tags around 22
            score = min(score,-0.02*num_tags_if_paired)
        if score > best:
            best = score
            best_next_ptr = j
        cnt += 1
    vertical_photos.append(vertical_tmp[best_next_ptr])
    vertical_tmp = vertical_tmp[:best_next_ptr] + vertical_tmp[best_next_ptr+1:]


# In[ ]:


# match vertical photos by tag length
idxs_list = [(a,b) for a,b in zip(vertical_photos[0::2], vertical_photos[1::2])]
idxs_list.extend([(a,) for a in horizontal_photos])
idxs_list.sort(key = lambda idxs: sum([len(pic_tags[idx]) for idx in idxs]))
calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# # ARRANGING combined/horizontal photos <a id='ID_ARRANGED'></a>

# In[ ]:


idxs_list_tmp = idxs_list
idxs_list = [idxs_list_tmp[0]]
idxs_list_tmp = idxs_list_tmp[1:]

for i in tqdm(range(len(idxs_list_tmp))):
    idxs1 = idxs_list[-1]
    best = -1
    best_next_ptr = -1
    cnt = 0
    for j,idxs2 in enumerate(idxs_list_tmp):
        if cnt > ARRANGE_WINDOW:
            # early stopping in the greedy search
            break
        if best == sum(len(pic_tags[idx]) for idx in idxs2)//2:
            # if we have reached the maximum possible score for the next neighbour
            break
        score = calc_idxs_pair_score(idxs1,idxs2)
        if score > best:
            best = score
            best_next_ptr = j
        cnt += 1
    idxs_list.append(idxs_list_tmp[best_next_ptr])
    idxs_list_tmp = idxs_list_tmp[:best_next_ptr] + idxs_list_tmp[best_next_ptr+1:]


# In[ ]:


calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


# # Solution analysis

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
                       ha='center', va='bottom', fontsize=8)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


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
        "maximum possible scores for vertical-vertical neighbours",
        "maximum possible scores for horizontal-vertical neighbours",
        "maximum possible scores for horizontal-horizontal neighbours"], 
         stacked=True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()


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


# # Write submission

# In[ ]:


calc_sequence(idxs_list), calc_sequence_max(idxs_list), len(idxs_list)


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

# In[ ]:




