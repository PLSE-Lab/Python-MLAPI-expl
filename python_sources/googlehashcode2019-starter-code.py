#!/usr/bin/env python
# coding: utf-8

# ## HashCode2019

# In[ ]:


from collections import OrderedDict
import numpy as np

def score(slides, photos):
    """Compute the score.
    Parameters
    -----------        
        slides : list of tuples
            Each item of the list is the tuple of photos ids
        
        photos : dict
            Dict of photos : {"id": {"orientation": "V/H", "tags": ["tag1", "tag2"]}}
    """
    total_score = 0
    for prev, curr in zip(slides, slides[1:]):
        prev_tags = set()
        for phot_id in prev:
            tags = photos[phot_id]["tags"]
            prev_tags |= set(tags)
        
        curr_tags = set()
        for phot_id in curr:
            tags = photos[phot_id]["tags"]
            curr_tags |= set(tags)
        
        common_tags = curr_tags & prev_tags
        prev_curr = prev_tags - common_tags
        curr_prev = curr_tags - common_tags

        total_score += min(len(common_tags), len(prev_curr), len(curr_prev))
    return total_score


def detailed_score(slides, photos):
    """Compute the detailedscore.
    Parameters
    -----------        
        slides : list of tuples
            Each item of the list is the tuple of photos ids
        
        photos : dict
            Dict of photos : {"id": {"orientation": "V/H", "tags": ["tag1", "tag2"]}}
    """
    scores = []
    for prev, curr in zip(slides, slides[1:]):
        prev_tags = set()
        for phot_id in prev:
            tags = photos[phot_id]["tags"]
            prev_tags |= set(tags)
        
        curr_tags = set()
        for phot_id in curr:
            tags = photos[phot_id]["tags"]
            curr_tags |= set(tags)
        
        common_tags = curr_tags & prev_tags
        prev_curr = prev_tags - common_tags
        curr_prev = curr_tags - common_tags

        scores.append(min(len(common_tags), len(prev_curr), len(curr_prev)))
    return scores


def parse_input(file):
	with open(file) as f:
		lines = f.readlines()[1:]

	pictures = OrderedDict()
	for i in range(len(lines)):
		line = lines[i]
		orientation, _, *tags = line.split()
		pictures[i] = {
			"orientation": orientation,
			"tags": list(tags)
		}
	return pictures


# In[ ]:


def submission_file(slides, filename="submission.txt"):
    """Prepare the submission file
    Parameters
    -----------
        slides : list of tuples
            Each item of the list is the tuple of photos ids
    Example
    ---------
    >>> slides = [(0,), (1, 2)]  # First slide has photo 0, seconde 1 & 2
    >>> submission_file(slides)
    """
    with open(filename, "w") as f:
        nb_slides = len(slides)
        f.write(f"{nb_slides}\n")
        for item in slides:
            for photo_id in item:
                f.write(f"{photo_id} ") 
            f.write("\n")
    


# In[ ]:


import math
import sys
from random import shuffle

# from data import parse_input, score
# from submission import submission_file


def _baseline_algo(pictures):
	output  = []
	i_picture = 0
	while pictures.get(i_picture, False):
		if pictures[i_picture]["orientation"] == "H":
			output.append((i_picture,))
			i_picture += 1
		elif pictures[i_picture]["orientation"] == "V" and 		     pictures.get(i_picture + 1) and 		     pictures[i_picture + 1]["orientation"] == "V":
			output.append((i_picture, i_picture + 1))
			i_picture += 2
		else:
			# output.append((i_picture,))
			i_picture += 1
	return output


def random_(photos):
    """Comme son nom l'indique."""
    ids = list(photos.keys())
    shuffle(ids)

    slides = []
    prev_empty_v = None
    for phot_id in ids:
        if photos[phot_id]["orientation"] == "H":
            slides.append((phot_id,))
        else:
            if prev_empty_v is None:
                slides.append((phot_id,))
                prev_empty_v = len(slides) - 1
            else:
                slides[prev_empty_v] = (slides[prev_empty_v][0], phot_id)
                prev_empty_v = None
    
    if prev_empty_v is not None:
        slides = slides[:prev_empty_v] + slides[prev_empty_v + 1:]
    return slides



def bruteforce_random(file_input, fileoutput):
	pictures = parse_input(file_input)
	best_score = 0
	best_slides = None
	iterations=3000
	while iterations:
# 		print("iterations",iterations)        
		output = random_(pictures)
		curr_score = score(output, pictures)
		iterations-=1

		if curr_score > best_score:
			best_score = curr_score
			best_slides = output

			sys.stdout.write('\r')
			sys.stdout.write("Best score : " + str(curr_score))
			sys.stdout.flush()
			submission_file(output, "submission.txt")


def main(file_input, fileoutput):
    pictures = parse_input(file_input)
    output = random_(pictures)
    submission_file(output, "submission_" + fileoutput)
    print("Score for", fileoutput, ":", score(output, pictures))



filenames = ["a_example.txt", "b_lovely_landscapes.txt", "c_memorable_moments.txt", "d_pet_pictures.txt", "e_shiny_selfies.txt"]
filenames = ["d_pet_pictures.txt"]


if __name__ == "__main__":
    # for filename in filenames:
    #     main(f"../data/{filename}", filename)
    bruteforce_random("../input/hashcode-photo-slideshow/d_pet_pictures.txt", "d_pet_pictures.txt")


# In[ ]:




