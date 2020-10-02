#!/usr/bin/env python
# coding: utf-8

# **This kernel shows how to get 98.85% ciphertexts matched with the plaintext without any decryption.
# The only thing we need to know is which character for each difficulty corresponds to the plaintext space character.
# The most frequent character for each of difficulty 1, 2 and 3 is easily found and used as space.
#  "1" is space for difficulty 1 and "8" is space for difficulty 2 and 3.**

# In[ ]:


from collections import defaultdict
import csv
import os
import tarfile


# **Setting up a dictionary to convert directory for a file to a target**

# In[ ]:


target_dir = {"0": "alt.atheism",
             "1": "comp.graphics",
             "2": "comp.os.ms-windows.misc",
             "3": "comp.sys.ibm.pc.hardware",
             "4": "comp.sys.mac.hardware",
             "5": "comp.windows.x",
             "6": "misc.forsale",
             "7": "rec.autos",
             "8": "rec.motorcycles",
             "9": "rec.sport.baseball",
             "10": "rec.sport.hockey",
             "11": "sci.crypt",
             "12": "sci.electronics",
             "13": "sci.med",
             "14": "sci.space",
             "15": "soc.religion.christian",
             "16": "talk.politics.guns",
             "17": "talk.politics.mideast",
             "18": "talk.politics.misc",
             "19": "talk.religion.misc"}
dir_target = {v: k for k, v in target_dir.items()}


# **A function that translates all but one character to a dot and the spacechar to a space**

# In[ ]:


def str_to_pattern(s, spacechar):
    frm = ''.join([chr(i) for i in range(256)]) + spacechar
    to = '.' * 256 + ' '
    transtab = str.maketrans(frm, to)
    return s.translate(transtab)


# **A function that fixes the plaindata in almost the same way as I foound in a comment from RS Turley**

# In[ ]:


def fix_plaindata(data):
    data = data.replace('\r\n', '\n')
    data = data.replace('\r', '\n')
    data = data.replace('\n', '\n ')
    if data.endswith('\n '):
        data = data[:-1]
    return data


# **Here we read all the plaintexts and converts them into pattern blocks.**

# In[ ]:


def get_plain_block_patterns():
    tar_fname = "../input/sklearn-20newsgroup/20news-bydate/20news-bydate.tar.gz"
    tar = tarfile.open(tar_fname, "r:gz")
    plain_block_pattern = defaultdict(list)
    for member in tar.getmembers():
        if member.isdir():
            continue
            
        # split filename and convert it to a target
        head, tail = os.path.split(member.path)
        _dir = os.path.split(head)[1] 
        target = dir_target[_dir]
        
        # read plaindata
        fh=tar.extractfile(member)
        plaindata = fh.read()
        plaindata = plaindata.decode('latin-1')
        plaindata = fix_plaindata(plaindata)
        
        # Split plaindata into chunks of length 300
        # and convert it to block patterns
        for pos in range(0, len(plaindata), 300):
            block = plaindata[pos:pos+300]
            pattern = str_to_pattern(block, ' ')
            plain_block_pattern[pattern].append(target)
    return plain_block_pattern
    


# **A function to read a csv file**

# In[ ]:


def get_test_rows():
    with open("../input/20-newsgroups-ciphertext-challenge/test.csv", 'r') as fh:
        reader = csv.reader(fh, delimiter=",")
        rows = []
        for i, row in enumerate(reader, 1):
            rows.append(row)
    return rows


# **The main function that classifies our ciphertexts**

# In[ ]:


def classify_by_pattern():
    print("Starting classification of difficulty 1,2 and 3")
    rows = get_test_rows()

    cipher_block_pattern = {}
    for _id, difficulty, ciphertext in rows:
        if difficulty == "1":
            s = str_to_pattern(ciphertext, "1")
            cipher_block_pattern[_id] = s
        elif difficulty in "23":
            s = str_to_pattern(ciphertext, "8")
            cipher_block_pattern[_id] = s

    plain_block_patterns = get_plain_block_patterns()
    correct, multiclass, noclass = 0, 0, 0
    classified_id_to_target = {}
    unclassified_id_to_target = {}
    for _id, cipher_pattern in cipher_block_pattern.items():
        if cipher_pattern in plain_block_patterns:
            targets = plain_block_patterns[cipher_pattern]
            target = targets[0]
            if all(t == target for t in targets):
                correct += 1
                classified_id_to_target[_id] = target
            else:
                multiclass += 1
                targets = set(int(t) for t in targets)
                unclassified_id_to_target[_id] = [str(t) for t in sorted(targets)]
        else:
            noclass += 1
    print("No class: %d   Correct: %d   Multi: %d" % (noclass, correct, multiclass))
    print("Success of classification %5.2f pct" % (100*correct/(correct+multiclass)))

    return classified_id_to_target, unclassified_id_to_target


# **Let's run our classification code and print some samples of the classified and unclassified ids.**

# In[ ]:


classified_id_to_target, unclassified_id_to_target = classify_by_pattern()

print("Sample of 5 classified ids")
for i, (_id, target) in enumerate(classified_id_to_target.items()):
    print(_id, target)
    if i > 5:
        break
print()

print("Sample of 5 unclassified ids")
for i, (_id, targets) in enumerate(unclassified_id_to_target.items()):
    print(_id, targets)
    if i > 5:
        break

