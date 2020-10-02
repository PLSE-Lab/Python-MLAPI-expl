import pandas as pd
import numpy as np
import collections

lyrics_df = pd.read_csv('../input/billboard_lyrics_1964-2015.csv', encoding='latin-1')

years = set(lyrics_df['Year'])

noise_words = ['you', 'your', 'that', 'this', 'and', 'but', 'when', 'got', 'like', 'the', 'for']

for year in years:
    most_popular = lyrics_df[lyrics_df.Year == year][lyrics_df.Rank == 1]
    artist = str(most_popular['Artist'].values)
    lyrics = str(most_popular['Lyrics'].values).split()
    filtered_lyrics = [l for l in lyrics if len(l) >= 3 and l not in noise_words]
    unique_words, counts = np.unique(filtered_lyrics, return_counts=True)
    print('Year: {}'.format(year))
    print('Most popular song was: {}, by: {}'.format(str(most_popular['Song'].values), artist))
    print('It has {} unique words'.format(len(unique_words)))

    counter = collections.Counter(filtered_lyrics)
    most_common_words = counter.most_common()
    print('The most common word in this song is: {}, appearing {} times.'.format(most_common_words[0][0], most_common_words[0][1]))

    all_entries = lyrics_df[lyrics_df.Year == year]
    all_lyrics = str(all_entries['Lyrics'].values).split()
    all_filtered_lyrics = [l for l in all_lyrics if len(l) >= 3 and l not in noise_words]
    all_counter = collections.Counter(all_filtered_lyrics)