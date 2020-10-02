#!/usr/bin/env python
# coding: utf-8

# Wanted to find out how much of Muller's report is redacted. 
# 
# Built using openCV and scikit image. Code identifies the redacted blocks of text using grayscale conversion and applying threshold for brightness. Redacted areas pop out as dark regions after erode / dilate operations. 
# 
# This notebook is for a sample page : Page 70

# In[2]:


# import the necessary packages
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt

from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

# load the image, convert it to grayscale, and blur it
image = cv2.imread("../input/report-070.ppm.jpg")
print("Original Image")
plt.imshow(image)
plt.show()

print("Grayscale Image")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

blurred = cv2.GaussianBlur(gray, (11, 11), 0)
print("Blurred Image")
plt.imshow(blurred)
plt.show()

# threshold the image to reveal redacted regions in the image
thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

print("Redactions highlighted")
plt.imshow(thresh)
plt.show()

labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

totalRedactedArea = 0

# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label != 0:
        continue

    # otherwise, construct the label mask and count the
    # number of pixels
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    # print(numPixels)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)

# find the contours in the mask, then sort them from left to right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image
    (x, y, w, h) = cv2.boundingRect(c)
    print("Redacted Zone:", i, "Area: ", w*h)
    totalRedactedArea = totalRedactedArea + w*h
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 5)
    cv2.putText(image, "#{}".format(w*h), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



(height, width, channels) = image.shape
cX = 210
cY = 210

cv2.rectangle(image, (cX,cY), (width-cX,height-cY), (0,255,0), 5)
# show the output image
print("Final Result")
plt.imshow(image)
plt.show()

totalArea = (width-cX) * (height-cY)
pctRedacted = (totalRedactedArea / totalArea)*100

print("Summary:")
print("Total Area of Document:", totalArea)
print("Total redacted Area:", totalRedactedArea)
print("%Redaction:",pctRedacted)


# Results across all pages:
# Page	Redaction %
# 0	0.03%
# 1	0.00%
# 2	0.00%
# 3	1.41%
# 4	0.00%
# 5	0.00%
# 6	4.67%
# 7	0.00%
# 8	0.00%
# 9	0.00%
# 10	0.00%
# 11	11.25%
# 12	5.52%
# 13	0.00%
# 14	0.00%
# 15	0.00%
# 16	14.19%
# 17	0.00%
# 18	0.00%
# 19	0.73%
# 20	0.00%
# 21	5.80%
# 22	26.27%
# 23	47.68%
# 24	52.85%
# 25	59.45%
# 26	47.91%
# 27	62.80%
# 28	38.93%
# 29	24.36%
# 30	54.32%
# 31	42.77%
# 32	1.78%
# 33	25.52%
# 34	21.12%
# 35	23.34%
# 36	7.20%
# 37	73.25%
# 38	26.03%
# 39	27.63%
# 40	28.89%
# 41	8.18%
# 42	11.94%
# 43	6.89%
# 44	9.08%
# 45	10.90%
# 46	41.19%
# 47	18.54%
# 48	4.53%
# 49	1.15%
# 50	2.79%
# 51	9.40%
# 52	3.26%
# 53	13.57%
# 54	15.18%
# 55	16.44%
# 56	7.20%
# 57	16.37%
# 58	33.21%
# 59	28.67%
# 60	44.89%
# 61	31.83%
# 62	44.62%
# 63	44.23%
# 64	54.78%
# 65	29.82%
# 66	30.24%
# 67	0.28%
# 68	0.00%
# 69	0.00%
# 70	1.42%
# 71	0.00%
# 72	8.16%
# 73	0.00%
# 74	3.51%
# 75	1.20%
# 76	1.05%
# 77	0.57%
# 78	0.00%
# 79	1.36%
# 80	0.00%
# 81	0.00%
# 82	0.45%
# 83	0.43%
# 84	0.00%
# 85	0.00%
# 86	0.00%
# 87	0.00%
# 88	0.00%
# 89	0.57%
# 90	8.57%
# 91	0.00%
# 92	3.85%
# 93	16.09%
# 94	0.00%
# 95	0.50%
# 96	0.00%
# 97	0.66%
# 98	28.03%
# 99	0.00%
# 100	9.06%
# 101	3.79%
# 102	0.77%
# 103	9.87%
# 104	4.31%
# 105	7.61%
# 106	0.53%
# 107	3.79%
# 108	26.01%
# 109	7.63%
# 110	5.79%
# 111	0.51%
# 112	0.00%
# 113	0.00%
# 114	0.00%
# 115	0.00%
# 116	6.34%
# 117	0.99%
# 118	20.51%
# 119	3.16%
# 120	0.55%
# 121	6.82%
# 122	0.00%
# 123	1.71%
# 124	24.78%
# 125	10.82%
# 126	0.60%
# 127	20.20%
# 128	15.32%
# 129	5.98%
# 130	0.60%
# 131	0.00%
# 132	0.00%
# 133	0.44%
# 134	0.00%
# 135	0.00%
# 136	0.00%
# 137	6.35%
# 138	0.00%
# 139	0.00%
# 140	0.00%
# 141	1.13%
# 142	0.01%
# 143	5.67%
# 144	5.64%
# 145	0.54%
# 146	2.36%
# 147	16.72%
# 148	0.00%
# 149	1.23%
# 150	30.58%
# 151	3.08%
# 152	0.00%
# 153	4.35%
# 154	4.46%
# 155	23.50%
# 156	26.84%
# 157	6.91%
# 158	10.47%
# 159	5.79%
# 160	13.71%
# 161	27.13%
# 162	8.16%
# 163	0.42%
# 164	0.00%
# 165	0.00%
# 166	0.00%
# 167	0.00%
# 168	0.56%
# 169	0.00%
# 170	2.26%
# 171	1.11%
# 172	5.31%
# 173	21.05%
# 174	10.19%
# 175	0.90%
# 176	0.42%
# 177	1.74%
# 178	0.00%
# 179	0.89%
# 180	0.00%
# 181	4.85%
# 182	0.00%
# 183	34.87%
# 184	68.21%
# 185	64.84%
# 186	50.14%
# 187	3.97%
# 188	0.00%
# 189	0.00%
# 190	10.60%
# 191	0.83%
# 192	0.00%
# 193	0.00%
# 194	0.00%
# 195	22.13%
# 196	57.06%
# 197	48.10%
# 198	23.50%
# 199	0.00%
# 200	0.00%
# 201	23.39%
# 202	0.00%
# 203	20.29%
# 204	52.54%
# 205	0.00%
# 206	47.18%
# 207	0.04%
# 208	0.00%
# 209	0.00%
# 210	0.70%
# 211	0.00%
# 212	0.00%
# 213	0.00%
# 214	0.59%
# 215	0.00%
# 216	0.00%
# 217	4.93%
# 218	0.00%
# 219	0.00%
# 220	0.00%
# 221	0.00%
# 222	0.00%
# 223	0.00%
# 224	4.70%
# 225	0.00%
# 226	0.29%
# 227	0.00%
# 228	6.03%
# 229	6.73%
# 230	0.00%
# 231	0.00%
# 232	0.00%
# 233	0.00%
# 234	0.00%
# 235	0.00%
# 236	0.00%
# 237	0.00%
# 238	0.00%
# 239	0.00%
# 240	0.00%
# 241	0.00%
# 242	0.00%
# 243	0.00%
# 244	0.00%
# 245	0.00%
# 246	0.00%
# 247	0.00%
# 248	0.00%
# 249	0.00%
# 250	0.00%
# 251	0.00%
# 252	0.00%
# 253	0.00%
# 254	0.00%
# 255	0.00%
# 256	0.00%
# 257	0.83%
# 258	0.00%
# 259	0.00%
# 260	0.00%
# 261	0.00%
# 262	0.00%
# 263	0.35%
# 264	0.00%
# 265	0.00%
# 266	0.00%
# 267	0.00%
# 268	0.00%
# 269	0.00%
# 270	0.00%
# 271	0.00%
# 272	0.00%
# 273	0.00%
# 274	0.00%
# 275	0.00%
# 276	0.00%
# 277	0.00%
# 278	0.00%
# 279	0.00%
# 280	0.00%
# 281	0.00%
# 282	0.00%
# 283	0.00%
# 284	0.00%
# 285	0.00%
# 286	0.00%
# 287	0.00%
# 288	1.45%
# 289	0.00%
# 290	0.00%
# 291	4.50%
# 292	0.00%
# 293	0.00%
# 294	0.00%
# 295	0.00%
# 296	0.00%
# 297	0.00%
# 298	0.00%
# 299	0.00%
# 300	0.00%
# 301	0.00%
# 302	4.42%
# 303	0.00%
# 304	0.00%
# 305	0.00%
# 306	0.00%
# 307	0.00%
# 308	4.83%
# 309	0.00%
# 310	0.00%
# 311	0.00%
# 312	0.00%
# 313	0.00%
# 314	0.00%
# 315	0.00%
# 316	1.43%
# 317	0.00%
# 318	0.00%
# 319	0.00%
# 320	0.00%
# 321	0.00%
# 322	0.00%
# 323	0.00%
# 324	0.00%
# 325	0.00%
# 326	0.00%
# 327	0.00%
# 328	0.00%
# 329	0.00%
# 330	0.00%
# 331	3.59%
# 332	0.00%
# 333	0.00%
# 334	0.00%
# 335	0.00%
# 336	0.00%
# 337	0.00%
# 338	0.00%
# 339	26.01%
# 340	50.58%
# 341	57.02%
# 342	0.16%
# 343	6.63%
# 344	33.40%
# 345	0.00%
# 346	0.00%
# 347	0.00%
# 348	0.00%
# 349	0.00%
# 350	0.00%
# 351	0.00%
# 352	0.00%
# 353	0.00%
# 354	0.00%
# 355	0.00%
# 356	0.00%
# 357	1.89%
# 358	3.07%
# 359	0.00%
# 360	0.00%
# 361	0.00%
# 362	7.78%
# 363	0.00%
# 364	0.00%
# 365	0.00%
# 366	0.00%
# 367	0.00%
# 368	0.00%
# 369	0.00%
# 370	0.00%
# 371	0.00%
# 372	0.00%
# 373	0.00%
# 374	0.00%
# 375	0.00%
# 376	0.00%
# 377	0.00%
# 378	0.00%
# 379	0.00%
# 380	0.00%
# 381	0.00%
# 382	0.00%
# 383	0.00%
# 384	0.00%
# 385	0.00%
# 386	0.00%
# 387	0.00%
# 388	0.00%
# 389	0.00%
# 390	0.00%
# 391	0.00%
# 392	0.00%
# 393	0.00%
# 394	0.81%
# 395	0.00%
# 396	0.00%
# 397	0.00%
# 398	0.71%
# 399	0.00%
# 400	0.00%
# 401	0.00%
# 402	5.97%
# 403	0.00%
# 404	4.18%
# 405	0.00%
# 406	1.19%
# 407	0.00%
# 408	0.00%
# 409	3.61%
# 410	0.00%
# 411	0.00%
# 412	0.00%
# 413	0.00%
# 414	0.75%
# 415	0.00%
# 416	0.00%
# 417	9.59%
# 418	0.00%
# 419	0.00%
# 420	0.00%
# 421	0.00%
# 422	0.00%
# 423	0.00%
# 424	0.00%
# 425	0.00%
# 426	0.00%
# 427	0.00%
# 428	0.00%
# 429	0.00%
# 430	0.00%
# 431	0.00%
# 432	0.00%
# 433	0.00%
# 434	0.00%
# 435	0.00%
# 436	0.00%
# 437	0.00%
# 438	3.71%
# 439	0.00%
# 440	0.77%
# 441	0.00%
# 442	0.00%
# 443	2.40%
# 444	19.39%
# 445	28.66%
# 446	51.31%
# 447	15.56%![image.png](attachment:image.png)
# 
# 
# 

# **Conclusion**: Some pages are heavily redacted. Especially towards the end of the report. But the overall redaction % is 6.05% and it surprisingly low when compared the attention it is getting.

# In[ ]:





# Credits: Logic is similar but inverse of the sample from here - 
# https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
