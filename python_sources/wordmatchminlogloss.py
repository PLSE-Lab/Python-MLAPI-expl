# Two questions:
# 1. Can you improve on this benchmark?
# 2. Can you beat the score obtained by this example kernel?

from nltk.corpus import stopwords
import pandas as pd


def word_match_share(row):
    q1words = {}
    q2words = {}
    print (row)
    a=[]
    b=[]
    for word in str(row[0]).lower().split():
        if word not in stops:
            q1words[word] = 1
            a.append(word)
    for word in str(row[1]).lower().split():
        if word not in stops:
            q2words[word] = 1
            b.append(word)
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    str1 = set(a)
    str2 = set(b)
    r_new = float(len(str1 & str2)) / len(str1 | str2)
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    print ("r",R)
    print ("r_new",r_new)
    
    return 0.3691978530262930+R*0.8305914675499357358-0.3674467744357837740


test = pd.DataFrame.from_csv("../input/test.csv")
stops = set(stopwords.words("english"))
test["is_duplicate"] = test.apply(word_match_share, axis=1, raw=True)
test["is_duplicate"].to_csv("count_words_benchmark.csv", header=True)