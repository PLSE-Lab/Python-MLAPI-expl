# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":

    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file

text = pytesseract.image_to_string(gray)

'''
os.remove(filename)
n_boxes = len(text['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


'''
t = text.split()
print(t)
print("length", len(text))
print(text[2])
length = len(t)
i = 0
while i < length:

    word = t[i]
    if len(word) == 1:
        t.remove(word)
        print(word)
        length = len(t)
        continue
    '''
	for letter in word:
		if letter.isalnum():
			continue
		else :
			print("replacing letter",letter,"from word",word)
			word=word.replace(letter,'')
			print ("word",word)

	#t[i]=word
	'''

    i += 1
print(t)
#	#print(word0+" "+word1)
#	if word0=="REG." and word1=="DT." :
#		print("REGN. DT",t[i+2])
#		break
# print(text.find("REGN DT"))
# print(text.find("NAME"))
# print(text.find("ENO"))
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
