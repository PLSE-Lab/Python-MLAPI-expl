#!/usr/bin/env python
# coding: utf-8

# # **||Introduction||**
# 
# # ***Speech Recognition***
# 
# Speech recognition is the ability of a machine or program to identify words and phrases in spoken language and convert them to a machine-readable format. 
# 
# Rudimentary speech recognition software has a limited vocabulary of words and phrases, and it may only identify these if they are spoken very clearly. 
# 
# More sophisticated software has the ability to accept natural speech.

# ![](https://camo.githubusercontent.com/4be2981e58bbb27bbf7bb060072d81e8284efd0b/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f726561646d652d6173736574732f766f6963652d627574746f6e2e676966)

# # ||Important requirements before building Speech recognition||
# 
# 1.Picking a Python Speech Recognition Package.
# * apiai
# * assemblyai
# * google-cloud-speech
# * pocketsphinx
# * SpeechRecognition------But this is best and have more advance parameters in it.
# * watson-developer-cloud
# * wit
# 
# 
# 2.Create the Recognizer Class.
# 
# 3.Working With Audio Files.
# 
# 4.Using recognize_audio to Capture Data through voices.
# 
# 

# # **||Installation||**
# 
# 1. The easiest way to install this is using **pip** **install** **SpeechRecognition**.
# 
# 2. **PyAudio** (for microphone users)
# PyAudio is required if and only if you want to use microphone input (Microphone). PyAudio version 0.2.11+ is required, as earlier versions have known memory management bugs when recording from microphones in certain situations.
# 
# * On Windows, install PyAudio using Pip: execute **pip** **install**** pyaudio** in a terminal.
# 
# 

# # ||Coding using Actual voice||
# 
# 
# import speech_recognition as sr
# 
# 
# r = sr.Recognizer()
# 
# 
# with sr.Microphone() as source:
# 
#   
#      print("Speak Anything :")
#     
#      audio = r.listen(source)
#     
#    try:
#     
#        text = r.recognize_google(audio)
#         
#        print("You said : {}".format(text))
#         
#    except:
#     
#         print("Sorry could not recognize what you said")
#         
#         
# 
# 

# # ||Code using audio file||
# 
# 
# import speech_recognition as sr 
# 
#   
# AUDIO_FILE = ("Hello.Mp3") 
#   
# 
#   
# r = sr.Recognizer() 
#   
# with sr.AudioFile(AUDIO_FILE) as source: 
#     #reads the audio file. Here we use record instead of 
#     #listen 
#     
#     audio = r.record(source)   
#   
# try: 
#      
#      print("The audio file contains: " + r.recognize_google(audio)) 
#   
# except sr.UnknownValueError: 
#   
#       print("Google Speech Recognition could not understand audio") 
#   
# except sr.RequestError as e: 
#    
#        print("Could not request results from Google Speech  
#            
#            Recognition service; {0}".format(e)) 

# **VOTE UP IF YOU LIKE IT**
# 
# ![](https://i.gifer.com/Kufr.gif)
