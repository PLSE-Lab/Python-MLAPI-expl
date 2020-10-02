
"""
This file simulates the leaderboard overfitting for fake leaderboard labels.

Info: https://markv.nl/blag/efficient-overfitting-of-training-data-kaggle-bowl
"""

from itertools import product
from sys import stdout
from numpy import array, log, log2, arange, floor, ones, ceil, concatenate, clip, stack
from numpy.random import RandomState
from scipy.stats import logistic


def logloss(pred, tru, cnt=1):
	cpred = clip(pred, 1e-15, 1-1e-15)
	return - array(
		(1 - tru) * log(1 - cpred) +
		(tru) * log(cpred)
	).sum() / cnt


"""
Create an imaginary test set.
"""
sample_cnt = 198
rs = RandomState(123456789)  # change this to try another vector
leaderboard_labels = rs.randint(0, 2, size=sample_cnt)

"""
Determine the precision.
"""
score_resolution = 0.00001
max_sample_score = logloss(1, 0) / sample_cnt
info_per_submission = log2(max_sample_score / score_resolution)
bits_per_submission = int(floor(info_per_submission)) + 1

"""
Prepare predictions.
I initially used `exp`, but found that Oleg's use of `sigmoid` works better
"""
steps = arange(0, bits_per_submission)
test_probabilities = logistic.cdf(-1 * score_resolution * 2**steps * sample_cnt)

"""
Iterate over the submissions.
"""
print('Need {0:.0f} submissions with {1:d} bits each'.format(ceil(float(sample_cnt / bits_per_submission)), bits_per_submission))
result = []
for offset in arange(0, sample_cnt - 1, bits_per_submission):
	
	"""
	Construct total prediction vector.
	"""
	if offset + bits_per_submission > sample_cnt:
		bits_per_submission = sample_cnt - offset
		test_probabilities = test_probabilities[:bits_per_submission]
	prediction = 0.5 * ones(sample_cnt)
	prediction[offset:offset + bits_per_submission] = test_probabilities
	
	"""
	Get the score (what the leaderboard does).
	"""
	score = logloss(prediction, leaderboard_labels) / sample_cnt
	score = round(score, ndigits=5)  # round the score, which is what the leaderboard will do
	stdout.write('offset = {1:3d} ; score = {0:.6f} '.format(score, offset))
	
	"""
	Find all possible vectors that have this score.
	Here `product` creates the Cartesian product of the input, with the input being [0, 1] for
	  each position, so it returns every possible binary vector of the specified length.
	"""
	result.append([])
	test_score = score - (sample_cnt - bits_per_submission) * -log(0.5) / sample_cnt
	for combi in product(*[[0, 1] for s in test_probabilities]):
		try_prediction = array(combi)
		try_score = logloss(test_probabilities, try_prediction) / sample_cnt
		if abs(test_score - try_score) < score_resolution / 2:
			stdout.write('.')
			result[-1].append(try_prediction)
	stdout.write('\n')

"""
Analyze the resulting possibilities.
Print the first 100 solutions with their error. Only 1 solution when no collisions.
"""
solutions = tuple(concatenate(p) for p in product(*result))
print('{0:d} solutions'.format(len(solutions)))
print('target labels: \n', leaderboard_labels)  # true solution
for solution in solutions[:100]:
	print('solution found:\n', solution, '({0:d} mistakes)'.format(sum(abs(solution - leaderboard_labels))))


