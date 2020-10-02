# Imports
import json

# Open file and load as JSON.
f = open("../input/data.json", "r")
data = json.load(f)

# Set up an array of tags.
corpus = []

# Iterate through projects.
for element in data:
    
    # Get tags from this project.
    tags = element['tags']
    
    # Continue onwards if no tags exist.
    if tags is None:
        continue

    # Iterate through all tags.
    corpus.extend(tags)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

corpus = " ".join(corpus)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', max_words=50).generate(corpus)

plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

fig = plt.figure(1)
fig.savefig("wordcloud.png", dpi=900)
f.close()


