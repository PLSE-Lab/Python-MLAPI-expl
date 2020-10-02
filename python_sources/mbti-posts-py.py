import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier


def prettify(word):
	return ''.join(e for e in word if e.isalnum()).lower()

ban_list = ['www', 'http', 'gif', 'jpg']
def word_is_normal(word):
	if len(word) != len(prettify(word)):
		return False
	for bann in ban_list:
		if bann in word.lower():
			return False
	return True

mbtype_list, posts_list = [], []
with open("../input/mbti_1.csv", 'r') as data_file:
	header = data_file.readline()
	for row in data_file:
# 	for x in range(14):
# 		row = data_file.readline()
		mbtype = row[0:4]
		posts = row[7:].split('|||')
		if mbtype and posts:
			mbtype_list.append(mbtype)
			posts_list.append(posts)
		else: "err: failed in type and post parsing"


usrs_token_freq = defaultdict(lambda: defaultdict(lambda: 0))
# usrs_token_freq :: {a: {}, b: {}, ..}
# where each a,b,. is a single user's posts index in posts_list and the key: {} is their token freq list.
word_index = [] ## put all words in this list, index in this list corresponds to feature number
for posts_idx, posts in enumerate(posts_list):
	for post_idx, post in enumerate(posts):
		for word in map(lambda x: x.lower(), post.split(' ')):
			if word_is_normal(word):
				word = prettify(word)
				usrs_token_freq[posts_idx][word] +=1 
				if word not in word_index:
					word_index.append(word)

## align training data's features and mb_type in corresponding list/matrix
t_data, t_results = [], []
for idx in usrs_token_freq:
  t_data.append(usrs_token_freq[idx])
  t_results.append(mbtype_list[idx])
assert(len(t_data) == len(t_results))

usrs_feats = np.zeros((len(t_results), len(word_index)))
for row_idx, row in enumerate(t_data):
	for word_idx, word in enumerate(word_index):
		if word in row:
			usrs_feats[row_idx][word_idx] = row[word]

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(usrs_feats, t_results)
predicted_results = clf.predict(usrs_feats)

matches = 0
for inx, predicted_result in enumerate(predicted_results):
    if predicted_result == mbtype_list[inx]:
        matches+=1
print("matches: ", matches)
succ_rate = matches/len(predicted_results)
print("classifier success rate: ", succ_rate)