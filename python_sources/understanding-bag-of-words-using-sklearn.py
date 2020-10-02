
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

vectorizer = CountVectorizer()

string1 = "As he crossed toward the pharmacy at the corner he involuntarily turned his head because of a burst of light that had ricocheted from his temple, and saw, with that quick smile with which we greet a rainbow or a rose, a blindingly white parallelogram of sky being unloaded from the van—a dresser with mirrors across which, as across a cinema screen, passed a flawlessly clear reflection of boughs sliding and swaying not arboreally, but with a human vacillation, produced by the nature of those who were carrying this sky, these boughs, this gliding façade."
string2 = "On offering to help the blind man, the man who then stole his car, had not, at that precise moment, had any evil intention, quite the contrary, what he did was nothing more than obey those feelings of generosity and altruism which, as everyone knows, are the two best traits of human nature and to be found in much more hardened criminals than this one, a simple car-thief without any hope of advancing in his profession, exploited by the real owners of this enterprise, for it is they who take advantage of the needs of the poor."
string3 = "My very photogenic mother died in a freak accident (picnic, lightning) when I was three, and, save for a pocket of warmth in the darkest past, nothing of her subsists within the hollows and dells of memory, over which, if you can still stand my style (I am writing under observation), the sun of my infancy had set: surely, you all know those redolent remnants of day suspended, with the midges, about some hedge in bloom or suddenly entered and traversed by the rambler, at the bottom of a hill, in the summer dusk; a furry warmth, golden midges."

phrases=[string1,string2,string3]

bag_of_words = vectorizer.fit(phrases)
bag_of_words = vectorizer.transform(phrases)

print(bag_of_words) # gives the sequence and the count of all words in order
print(vectorizer.vocabulary_.get("the")) # find the word "the" form the bag of words

sw = stopwords.words("english") # to know the list of stow words (words with less meaning or specificty)

print(len(sw)) #Total 179 stop words are there in english according to the nltk package

## Stemming is the concept in which the similar words are grouped to the root word.
## Example:  response, responsivity, responsiveness, respons will be grouped to the root word "respons"

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print(stemmer.stem("responsiveness")) # gives root words respons
print(stemmer.stem("responsivity"))   # gives root words respons    
print(stemmer.stem("response"))       # gives root words respons     
