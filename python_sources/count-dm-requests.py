""" A kernel posted on Kaggle that shows how to pull just the first consumer request and
    company response from the dataset.
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

tweets = pd.read_csv('../input/twcs/twcs.csv')


# Pick only inbound tweets that aren't in reply to anything...
first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
print('Found {} first inbound messages.'.format(len(first_inbound)))

# Merge in all tweets in response
inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')
print("Found {} responses.".format(len(inbounds_and_outbounds)))

# Filter out cases where reply tweet isn't from company
inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]

# Et voila!
print("Found {} responses from companies.".format(len(inbounds_and_outbounds)))
print("Tweets Preview:")
print(inbounds_and_outbounds)


print((inbounds_and_outbounds['text_y'].str.contains(' DM') | inbounds_and_outbounds['text_y'].str.contains('https://t.co')).sum() / len(inbounds_and_outbounds))
