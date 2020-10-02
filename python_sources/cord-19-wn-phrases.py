from nltk.corpus import wordnet as wn

"""
Clean clean then clean the code
"""
def wn_phrases(paper):
    for si, sent in enumerate(paper):
        sent = sent.text

        new_sent = []
        n = len(sent)
        i = 0
        look_ahead = 4
        while i<n:
            j = min(look_ahead, max(0, n-i-1))
            while j>0:
                ngram = '_'.join(sent[i:i+j+1])
                if len(wn.synsets(ngram)):
                    # Found a match
                    break
                j-=1

            assert j>=0

            if j==0:
                new_sent.append(sent[i])
            else:
                new_sent.append(ngram)

            i += 1+j

        paper.sentences[si].text = new_sent
