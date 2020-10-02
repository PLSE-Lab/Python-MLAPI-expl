#!/usr/bin/env python
# coding: utf-8

# # Efficient Sentiment and Metadata extractor
# Some of you (like me :D) might have forked [BaselineModeling](https://www.kaggle.com/wrosinski/baselinemodeling) or any other kernel which origins from that kernel. This kernel loads metadata/sentiments by utilizing the class `PetFinderParser` through the function `extract_additional_features`. Depending on how many CPU cores available, this might take up to 30 min which is quite slow. Most of the slowdown is caused by a sorting function in `extract_additional_features`. I replaced it without sorting which gives a speed up by at least a factor of 5. Feel free to use it :). 

# In[ ]:


def extract_additional_features(pet_id, mode='train'):
    
    sentiment_filename = f'../input/petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json'
    try:
        sentiment_file = pet_parser.open_json_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    for ind in range(1,200):
        metadata_filename = '../input/petfinder-adoption-prediction/{}_metadata/{}-{}.json'.format(mode, pet_id, ind)
        try:
            metadata_file = pet_parser.open_json_file(metadata_filename)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        except FileNotFoundError:
            break
    if dfs_metadata:
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]
    return dfs


# In[ ]:




