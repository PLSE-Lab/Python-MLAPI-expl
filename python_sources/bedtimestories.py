# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import os

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as shellcityisagiftshop

from keras.models import load_model

# Auto
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""   
t = data
clean_t = ""
for c in t:
    if c.isalpha():
        clean_t += c.lower()
    elif c == "\n":
        clean_t += "\n"
    elif c == "'":
        clean_t += ""
    else:
        clean_t += " "
data = clean_t

# Story Generation Code
tokenizer = Tokenizer()

corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_seq = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_seq.append(n_gram_seq)

max_seq_len = max([len(x) for x in input_seq])
input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding = 'pre'))

prev_words, next_word = input_seq[:,:-1],input_seq[:,-1]
next_word = shellcityisagiftshop.to_categorical(next_word, num_classes=total_words)

#### Creating Model
input_len = max_seq_len - 1

model = Sequential()
model.add(Embedding(total_words, 10, input_length = input_len))
model.add(LSTM(180))
model.add(Dropout(0.5))
model.add(Dense(total_words, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#model = load_model('ghost_story_generator.h5')

model.fit(prev_words, next_word, epochs = 40, verbose = 1)

def generate_text(model, seed_text, next_words, max_seq_len):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= 
                             max_seq_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word + "\n"
    return seed_text

#### Testing
print(generate_text(model, "I was was going to the bathroom and I saw a ", 100, max_seq_len))

"""
#### Saving
#model.save("ghost_story_generator")


# Dataset of stories
data = """
The Cell Phone

A couple of months ago, my friend's cousin (a single mother) bought a new cell phone. After a long day of work, she came home, placed her phone on the counter, and went watch to TV; her son came to her and asked if he could play with her new phone. She told him not to call anyone or mess with text messages, and he agreed.
At around 11:20, she was drowsy, so she decided to tuck her son in and go to bed. She walked to his room and saw that he wasn't there. She then ran over to her room to find him sleeping on her bed with the phone in his hand.
Relieved, she picked her phone back up from his hand to inspect it. Browsing through it, she noticed only minor changes such as a new background, banner, etc., but then she opened up her saved pictures. She began deleting the pictures he had taken, until only one new picture remained.
When she first saw it, she was in disbelief. It was her son sleeping on her bed, but the picture was taken by someone else above him... and it showed the left half of an elderly woman's face.

The Rocking Horse

One night, when I was maybe 10-12, I had trouble falling asleep. My bedroom was the entire top floor of our house with my bed and such being on the left side and storage closets and a play area being on the right. I was lying in bed when I heard a noise from the other side of the room and see a rocking horse begin to rock. It was sitting just outside one of the storage closet doors. It proceeded to rock its way halfway across the room and stopped dead under the ceiling light. At this point I was freaking out and just buried my head under my blankets and never peeked out again until morning.
It was all confirmed to not be a dream as the rocking horse was still in the middle of my room when I woke up. Furthermore, I got a stern reprimand from my parents for being up out of bed playing with my toys well past my bedtime. Their bedroom was directly below the storage closet/play area and had heard the creaking of the rocking horse shuffling across the room.

The Following

My older sister has a ghost that's followed her around for years.
I lived with her once for about 3 months, and so much weird stuff happened in that time. All my sister would say to me when I mentioned it was that her ghost "didn't like me being there."
Things like going to bed with everything locked up and switched off and waking up in the morning with the back door open, lights on and the kettle switched on. One night my sister and I were getting ready to go out and I'd asked to borrow her liquid foundation. I used it and put it back where she kept her makeup. Ten minutes later she's asking me for it and it was nowhere to be seen. She accused me of taking it and made me buy her a new one and refused to listen to my side of the story. About a year or so later when she was packing to move to a new house, she found the makeup in a shoebox with some old letters. The shoebox was in a zipped up suitcase that was underneath her bed.
But probably the most scared I ever felt was one afternoon when I was the only one in the house (which never happened as four other people lived there). I'd arrived home from work and headed straight to the bathroom. All the doors/windows etc were closed. I was standing in the bathroom and started squeezing a pimple on my chin when a female voice in the hall said "stop picking your zits!" It was loud enough and sounded real enough and at the time I thought it was my sister. So I laughed, told her to "fuck off" and asked what she was doing for dinner. No answer. I stuck my head out into the hall. No one there. I searched the house top to bottom and there was no one home. I sat out on the front porch until someone else got home because I didn't want to be in there alone.

The Grandfather

My grandfather told me this story about how one time he was sitting in a chair in front of the house, when he heard his wife repeatedly calling him from inside the house. The thing is, my grandmother passed away a few years before that. But he told me that the voice was so pressing that he actually got up to look inside the house, and as soon as he got inside he heard a loud crash behind him and turned around to see that the chair he has been sitting in moments ago had been crushed by the cast iron gutter that fell on it. If he hadn't come inside the house he would have probably been seriously injured. I don't know if it's paranormal or not, but every time I think about it it sends chills down my spine.

The Shadow

I didn't know that's what it was called until much later. I was living in a house in Laguna Beach that had been there since the 1920s. In it's history, it had been a speakeasy, a brothel and a house for smuggling illegal immigrants.
One day, my new wife and I were having an argument. I can't even recall what it was about. She walked down the block to get a cup of coffee and cool off, and I was alone in the house. The way the place was built was incredibly haphazard. There was a bedroom and living room on one side, then a bathroom with two entrances. On the other side of the bathroom was a hallway that had windows in one side and two bedrooms on the other. From my bedroom, I could look across the hall into the bathroom, then through the bathroom and down the other hall. I was standing at my dresser, and I just noticed movement out the corner of my eye, and looked down there. There was... and honest to god, this gives me goose bumps just typing it, 17 years later, a black figure. It was maybe three feet tall, and it was only vaguely humanoid. it looked like black scribbles, like someone had scribbled a human shape, but the scribbles moved, like electricity arcing, that's the best way to describe it.
There was no sound that I could remember. I distinctly remember when I saw it I wasn't afraid, just like, WTF? Then it noticed me looking at it. I can't say it turned around, it just, focused on me I guess. THEN I was scared. I didn't move, didn't scream, nothing, I was just frozen, because it just fucking came at me, it RUSHED down the hall towards me. I have no idea what it intended, but as soon as it entered the bathroom, the door closest to me just SLAMMED shut on it. I screamed. I yelled for my wife. She wasn't home. I went the fuck outside, into the daylight, and didn't go back in until she got home about 10 minutes later.
I don't believe in ghosts. I don't believe I saw something supernatural, but I know I saw something. I don't know what it was.
"""