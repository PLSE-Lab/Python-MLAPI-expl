#!/usr/bin/env python
# coding: utf-8

# # Text-to-Speech(TTS) 
# 
# Text to speech, abbreviated as TTS, is a form of speech synthesis that converts text into spoken voice output.
# 
# Text to speech systems were first developed to aid the visually impaired by offering a computer-generated spoken voice that would "read" text to the user.
# 
# TTS should not be confused with voice response systems. Voice response systems synthesize speech by concatenating sentences from a database of prerecorded words and are used for different purposes than TTS systems, which form sentences and/or phrases based on a language's graphemes and phonemes.
# 
# Voice response systems are limited to synthesizing sentences that contain only words that have been predetermined by the system.
# 
# TTS systems, in contrast, are theoretically capable of "reading" any string of text characters to form original sentences. turned into audio.
# 
# 

# ![](https://i2.wp.com/haptik.ai/blog/wp-content/uploads/2018/09/Blog-Image-06-09-2018.gif?resize=768%2C402&ssl=1)

# # conda install gTTS

#         
# from gtts import gTTS
# 
# text=""
# 
# #You can use directly file or you can write text file
# 
# 
# 
# #text="Hello guys please vote up"
# 
# 
# 
# with open("New Text Document.txt","r") as file:
#      
#      for line in file:
#         text=text + line
# 
# 
# speech=gTTS(text,lang = 'hi',slow = True ,lang_check = True)
# 
# speech.save("hello.mp3")
