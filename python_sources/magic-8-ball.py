# I can't figure out how you can run the code when its online
# I am a beginner, I don't know too much about this
# I guess you have to copy and paste it into another editor?
# I don't know

# I did none of the code below
# I just wanted to share some of the fun code I found in "AUTOMATE BORING STUFF WITH PYTHON"
# Upvote if you had fun!

# Thanks to:
# AUTOMATE BORING STUFF WITH PYTHON by AL SWEIGART
# For all the code
# Wikipedia on all Magic 8-ball responses
# Link: https://en.wikipedia.org/wiki/Magic_8-Ball

import random

def obtainAnswer(AnswerThingy):
    if AnswerThingy == 1:
        print('It is certain')
    elif AnswerThingy == 2:
        print('It is decidedly so')
    elif AnswerThingy == 3:
        print('Without a doubt')
    elif AnswerThingy == 4:
        print('Yes - definitely')
    elif AnswerThingy == 5:
        print('You may rely on it')
    elif AnswerThingy == 6:
        print('As I see it, yes')
    elif AnswerThingy == 7:
        print('Most likely')
    elif AnswerThingy == 8:
        print('Outlook good')
    elif AnswerThingy == 9:
        print('Yes')
    elif AnswerThingy == 10:
        print('Signs point to yes')
    elif AnswerThingy == 11:
        print('Reply hazy, try again')
    elif AnswerThingy == 12:
        print('Ask again later')
    elif AnswerThingy == 13:
        print('Better not tell you now')
    elif AnswerThingy == 14:
        print('Cannot predict now')
    elif AnswerThingy == 15:
        print('Concentrate and ask again')
    elif AnswerThingy == 16:
        print("Don't count on it")
    elif AnswerThingy == 17:
        print('My reply is no')
    elif AnswerThingy == 18:
        print('My sources say no')
    elif AnswerThingy == 19:
        print('Outlook not so good')
    elif AnswerThingy == 20:
        print('Very doubtful')

r = random.randint(1, 20)
Answer = obtainAnswer(r)
print(Answer)


              
        