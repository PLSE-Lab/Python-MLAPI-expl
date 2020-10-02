# %% [code]
import re
import json

DEBUG = False

class Match(object):

    def __init__(self):
        self.trie = {}

    def match(self, ctx):
        ctx = ctx.lower()
        words = re.split('\s+', ctx)
        A = []
        i = 0
        while (i < len(words)):
            j = self.match2(words, i, self.trie)
            if j >= 0:
                A.append('_'.join(words[i:i+j+1]))
                i += j
            else:
                A.append(words[i])
            i += 1
        return ' '.join(A)

    def match_offsets(self, ctx):
        ctx = ctx.lower()
        words = re.split('\s+', ctx)
        offsets = []
        i = 0
        while (i < len(words)):
            j = self.match2(words, i, self.trie)
            if j >= 0:
                offsets.append((i,i+j+1))
                i += j
            i += 1
        return offsets

    def match2(self, words, i, trie):
        if words[i] not in trie:
            return -1
        for length in sorted(trie[words[i]].keys(), reverse=True):
            context = ' '.join(words[i+1:i+length+1])
            for entry in trie[words[i]][length]:
                if context == entry:
                    return length
        return -1

    def matchinit(self, dictfile):
        f = open(dictfile, 'r')
        for line in f:
            entry = re.split('\s+',line)[0]
            firstword = re.split('_+', entry)[0]
            rwords = re.split('_+', entry)[1:]

            length = len(rwords)
            if firstword not in self.trie:
                self.trie[firstword] = {}
            if length not in self.trie[firstword]:
                self.trie[firstword][length] = []
            self.trie[firstword][length].append(' '.join(rwords))
        f.close()

    def matchinit_from_list(self, entrylist):
        self.trie = {}  # destroy previous init
        for entry in entrylist:
            firstword = re.split(' ', entry)[0]
            rwords = re.split(' ', entry)[1:]

            length = len(rwords)
            if firstword not in self.trie:
                self.trie[firstword] = {}
            if length not in self.trie[firstword]:
                self.trie[firstword][length] = []
            self.trie[firstword][length].append(' '.join(rwords))


class LabeledMatch(object):
    def __init__(self):
        self.trie = {}
        self.labels = {}

    def match(self, ctx):
        #ctx = ctx.lower()
        words = re.split('\s+', ctx)
        A = []
        i = 0
        while i < len(words):
            j = self.match2(words, i, self.trie)
            if j >= 0:
                if DEBUG: print("i:",str(i), "word:", ' '.join(words[i:i+j+1]))
                span = ' '.join(words[i:i+j+1])
                labels = '|'.join(self.labels[span.lower()])
                A.append(span + '|' + labels)
                i += j
            else:
                A.append(words[i] + '|O')
            i += 1
        return ' '.join(A)

    def matchjson(self, json_sent):
        words = [wf['token'] for wf in json_sent['annotatedSentence']['sentence']]
        i = 0
        while i < len(words):
            j = self.match2(words, i, self.trie)
            if j == 0:
                json_sent['annotatedSentence']['sentence'][i]['label'] = '|'.join(['B-' + label for label in self.labels[words[i].lower()]])
            elif j > 0:
                if DEBUG: print("i:",str(i), "word:", ' '.join(words[i:i+j+1]))
                span = ' '.join(words[i:i+j+1])
                for k in range(j+1):
                    if k == 0:
                        prefix = 'B'
                    else:
                        prefix = 'I'
                    json_sent['annotatedSentence']['sentence'][i+k]['label'] = '|'.join([prefix+'-' + label for label in self.labels[span.lower()]])
                i += j
            else:
                json_sent['annotatedSentence']['sentence'][i]['label'] = 'O'
            i += 1
        return json_sent

    def match2(self, words, i, trie):
        wordi = words[i].lower()
        if wordi not in trie:
            return -1
        for length in sorted(trie[wordi].keys(), reverse=True):
            context = ' '.join(words[i+1:i+length+1]).lower()
            for entry in trie[wordi][length]:
                if context == entry:
                    return length
        return -1

    def matchinit(self, labeled_dict):
        self.trie = {}  # destroy previous init
        self.labels = {}
        for slot in labeled_dict:
            # TODO: This will change. Labeled_dict will be formatted in json.
            fillers = labeled_dict[slot]

            for filler in fillers:
                # store all possible labels for each filler
                filler = str(filler).lower().strip()
                if filler not in self.labels:
                    self.labels[filler] = []
                self.labels[filler].append(slot)

                # populate trie dict for matching
                if DEBUG: print('slot:', slot, '- filler:', filler)

                firstword = re.split(' ', filler)[0]
                rwords = re.split(' ', filler)[1:]

                length = len(rwords)
                if firstword not in self.trie:
                    self.trie[firstword] = {}
                if length not in self.trie[firstword]:
                    self.trie[firstword][length] = []
                self.trie[firstword][length].append(' '.join(rwords))


if __name__ == '__main__':
    vocab = ['virus transmission', 'incubation']
    sentence = "How long is the incubation after the virus transmission"
    matcher = Match()
    matcher.matchinit_from_list(vocab)
    matched_sentence = matcher.match(sentence)
    print(matched_sentence)