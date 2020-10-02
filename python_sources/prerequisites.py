# You can use this script to test your prerequisities for participating in the competition
success = True
try:
    import os
    import cv2
    import matplotlib
    import numpy
    import sklearn
except ImportError as e:
    print("You have not installed the necessary libraries.")
    print(e)
    success = False

if success:
    print("You're good to go! Good luck for the competition!")
    