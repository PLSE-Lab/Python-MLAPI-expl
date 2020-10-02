import sqlite3
import sys

sentences=set()


def is_good(sentence, ending):
    if len(sentence) > 75:
        return False
    if len(sentence) < 13:
        return False
    if not sentence[0].isupper():
        return False
    global sentences
    if (sentence in sentences): 
        return False
    if (ending == 'on.' and sentence.endswith('i' + ending)):
        return False
    if (ending[0]=='o' and sentence.endswith('o' + ending)):
        return False
    sentences.add(sentence)
    return True


def extract_sentence(words, ending):
    delims = [';','@','.',':','"',"*",'--','?','_',')','(']
    sentence = []
    for word in words:
        sentence.append(word)
        if word.endswith(ending):
            return ' '.join(sentence)
        if any(delim in word for delim in delims):
            sentence = []
            continue


def takePhrase(ending, min_id=0):
    con = sqlite3.connect('../input/database.sqlite')
    with con:
        while True:
            cur = con.cursor()    
            cur.execute(("SELECT Id, ExtractedBodyText FROM Emails WHERE"+
                         " ExtractedBodyText LIKE '%{}%' AND Id > {} "+
                         " order by Id LIMIT 1").format(ending, min_id))
            data = cur.fetchone()
            min_id  = data[0] 
            sentence = extract_sentence(data[1].split(), ending)
            if sentence == None:
                return min_id, sentence
            if is_good(sentence, ending):
                return min_id, sentence


def rhyme_generate(endings, total):
    min_id = 0;
    it = iter(endings)
    ending = next(it)
    for i in range(total):
            min_id, phrase = takePhrase(ending, min_id)
            while (phrase == None):
                ending = next(it)
                min_id, phrase = takePhrase(ending)
            yield phrase


def main():
    r1 = rhyme_generate(['on.','ly.','ion.','d.','.t'], 26)
    r2 = rhyme_generate(['ck.','.on','k.','ng.'], 26)
    for i in range(26):
        print(next(r1)[:-1])
        print(next(r2)[:-1])
            

if (__name__ == '__main__'):
	main()
