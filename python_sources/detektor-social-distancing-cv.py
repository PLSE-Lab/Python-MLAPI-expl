#!/usr/bin/env python
# coding: utf-8

# # Social Distancing Detektor Berbasis Computer Vision

# In[ ]:


# import library (pustaka) yang dibutuhkan
import cv2 #pustaka pengolah image dan video
import numpy as np #pustaka pengolah array
import matplotlib.pyplot as plt #pustaka untuk visualisasi

# install library tambahan (download)
get_ipython().system('pip install imutils')
import imutils

from scipy.spatial import distance as dist #pustaka u/ menghitung jarak
from collections import OrderedDict #pustaka u/ sorting
import time


# # Centroid Tracker
# Berfungsi untuk mentracking centroid (titik tengah/pusat) dari objek-
# yang dideteksi (orang).
# Berfungsi juga sebagai pendaftar ID (identitas) dari masing-
# masing objek.
# Berbasis pada kode Adrian Rosebrock, https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/. Dengan beberapa penyesuaian

# In[ ]:


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
    
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int(startX - (endX/2))
            cY = int(startY - (endY/2))
            inputCentroids[i] = (cX, cY)
            #cX = int((startX + endX) / 2.0)
            #cY = int((startY + endY) / 2.0)
            
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


# # Utills Class
# Utills Class atau kelas utilitas berisi fungsi yang berguna dalam membantu
# penerapan sistem seperti :
# 1. Transformasi dari prespective-view ke bird-view
# 2. Menghitung pelanggaran social distancing 

# In[ ]:


# kelas utills atau utilitas
class utills:

    # Fungsi u/ menentukan bottom center dari semua bounding boxes object--
    # --dan kemudian akan digunakan u/ melakukan transformasi dari prespective-view ke bird-view
    def get_transformed_points(boxes, prespective_transform):
        
        # initialize rects dan bottom_points yg berguna untuk menyimpan array bottom center--
        # --dari bounding box object
        rects = []
        bottom_points = []
        
        for box in boxes:
            pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]]] , dtype="float32")
            bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
            pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
            pnt_bird = [int(bd_pnt[0]), int(bd_pnt[1]), 0, 0]
            
            bottom_points.append(pnt)
            rects.append(np.array(pnt_bird))

        return bottom_points, rects
    
    # Fungsi u/ menghitung jarak antar dua point object (orang).
    # distance_w, distance_h mempresentasikan pixel-to-metric ratio atau--
    # --besar nilai pixel untuk jarak 180 cm dalam frame video.
    def cal_dis(p1, p2, distance_w, distance_h):

        h = abs(p2[1]-p1[1])
        w = abs(p2[0]-p1[0])

        dis_w = float((w/distance_w)*180)
        dis_h = float((h/distance_h)*180)

        return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

    # Fungsi u/ menghitung jarak antar semua titik object dan--
    # --menghiutng closeness ratio (rasio kedekatan).
    def get_distances(boxes1, bottom_points, distance_w, distance_h):

        distance_mat = []
        bxs = []

        for i in range(len(bottom_points)):
            for j in range(len(bottom_points)):
                if i != j:
                    dist = utills.cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
                    #dist = int((dis*180)/distance)
                    if dist <= 150:
                        closeness = 0
                        distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])
#                     elif dist > 150 and dist <=180:
#                         closeness = 1
#                         distance_mat.append([bottom_points[i], bottom_points[j], closeness])
#                         bxs.append([boxes1[i], boxes1[j], closeness])       
                    else:
                        closeness = 2
                        distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])

        return distance_mat, bxs

    # Function gives scale for birds eye view  
    # Fungsi memberikan skala untuk transformasi bird-view
    # Skala yg digunakan w:480, h:1180 (video=1080 + pad=100) 
    def get_scale(W, H):
        
        dis_w = 480
        dis_h = 1180
        
        return float(dis_w/W),float(dis_h/H)

    # Fungsi u/ menghitung jumlah objek (orang) yg melakukan pelanggaran
    def get_count(distances_mat):

        r = []
        g = []
        #y = []

        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 0:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    r.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    r.append(distances_mat[i][1])

#         for i in range(len(distances_mat)):

#             if distances_mat[i][2] == 1:
#                 if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
#                     y.append(distances_mat[i][0])
#                 if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
#                     y.append(distances_mat[i][1])

        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 2:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    g.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    g.append(distances_mat[i][1])

        #return (len(r),len(y),len(g))
        return (len(r), len(g))


# # Plot Class
# Plot Class atau Kelas Ploting digunakan u/ melakukan fungsi plotting atau drawing pada frame video

# In[ ]:


class plot:

    # Fungsi u/ melakukan transformasi bird-view
    def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count, objects):
        h = frame.shape[0]
        w = frame.shape[1]

        red = (0, 0, 255)
        green = (0, 255, 0)
        #yellow = (0, 255, 255)
        white = (200, 200, 200)
        black = (0,0,0)
        
        blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8)
        blank_image[:] = white
        warped_pts = []
        r = []
        g = []
        #y = []
        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 0:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    r.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    r.append(distances_mat[i][1])

                blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), 
                                                     int(distances_mat[i][0][1] * scale_h)), 
                                       (int(distances_mat[i][1][0] * scale_w), 
                                        int(distances_mat[i][1][1]* scale_h)), red, 2)

#         for i in range(len(distances_mat)):

#             if distances_mat[i][2] == 1:
#                 if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
#                     y.append(distances_mat[i][0])
#                 if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
#                     y.append(distances_mat[i][1])

#                 blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), yellow, 2)

        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 2:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    g.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    g.append(distances_mat[i][1])

        for i in bottom_points:
            blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 10)
#         for i in y:
#             blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, yellow, 10)
        for i in r:
            blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 10)

        # Tampilkan object ID pada setiap objek yg terdeteksi--
        # --pada frame bird-view
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(blank_image, text, (int(centroid[0] * scale_w) - 10, int(centroid[1] * scale_h) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return blank_image
    
    # Fungsi u/ drawing bounding boxes pada frame perspective view--
    # --dan drawing lines antar objek yg melakukan pelanggaran
    def social_distancing_view(frame, distances_mat, boxes, risk_count, bird_view):

        red = (0, 0, 255)
        green = (0, 255, 0)
        #yellow = (0, 255, 255)

        for i in range(len(boxes)):

            x,y,w,h = boxes[i][:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)
            
#         for i in range(len(distances_mat)):

#             per1 = distances_mat[i][0]
#             per2 = distances_mat[i][1]
#             closeness = distances_mat[i][2]

#             if closeness == 1:
#                 x,y,w,h = per1[:]
#                 frame = cv2.rectangle(frame,(x,y),(x+w,y+h),yellow,2)

#                 x1,y1,w1,h1 = per2[:]
#                 frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),yellow,2)

#                 frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),yellow, 2) 

        for i in range(len(distances_mat)):

            per1 = distances_mat[i][0]
            per2 = distances_mat[i][1]
            closeness = distances_mat[i][2]

            if closeness == 0:
                x,y,w,h = per1[:]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)

                x1,y1,w1,h1 = per2[:]
                frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)

                frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
                
            # buat pad (padding) pada sisi bawah frame prespective-view
            # dengan h=100, dan w=frame.shape[1]
            pad = np.full((100,frame.shape[1],3), [250, 250, 250], dtype=np.uint8)

            # draw text pada padding
            cv2.putText(pad, "Jumlah Orang Terdeteksi : " + str(risk_count[0] + risk_count[1]) + " Orang", (100, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 0), 2)
            cv2.putText(pad, "Jumlah Pelanggaran Social Distancing : " + str(risk_count[0]) + " Orang", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    
        # gabungkan pad dengan frame prespective-view
        # dan kemudian gabungkan dengan bird-view
        frame = np.vstack((frame,pad))
        frame = np.hstack((frame, bird_view))
    
        return frame


#         for (objectID, centroid) in objects.items():
#             text = "ID {}".format(objectID)
# #             centroid = np.array([[[centroid[0], centroid[1]]]] , dtype="float32")
# #             normal = cv2.perspectiveTransform(centroid, inv_trans)
# #             cv2.putText(blank_image, text, (int(normal[0][0][0]) - 10, int(normal[0][0][1]) + 20),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cv2.putText(blank_image, text, (int(centroid[0] * scale_w) - 10, int(centroid[1] * scale_h) - 15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# # Processing Social Distancing Detector

# In[ ]:


# Fungsi u/ melakukan processing social distancing detector
def calculate_social_distancing(vid_path, net, output_dir, output_vid, ln1, points):
    
    # initialize count dan video capture
    count = 0
    vs = cv2.VideoCapture(vid_path)    

    # Ambil video height, width dan fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
    # Tentukan skala untuk bird-view
    scale_w, scale_h = utills.get_scale(width, height)
    
    # initialize penyimpanan video output hasil processing
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_movie = cv2.VideoWriter("perspective_view.avi", fourcc, fps, (1464, 720), True)
    #bird_movie = cv2.VideoWriter("bird_eye_view.avi", fourcc, fps, (400, 600), True)
    
    #points = []
    
    global image
    
    # mulai processing dengan loop pada video capture
    while True:
        
        # ambil grab dan frame
        (grabbed, frame) = vs.read()

        # berhenti saat nilai grab = false
        if not grabbed:
            print("[INFO] Processing done...")
            break
        
        # ambil H dan W dari frame vs
        (H, W) = frame.shape[:2]
          
        # initialize src yg berisi 4 titik tranformasi, dan--
        # --dst yang berisi 4 titik ukuran vs sebenarnya
        # prespective_transform berisi matriks u/ transformasi prespective ke bird-view
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)
        
        # invers matriks u/ transformasi bird ke prespective-view
        inv_trans = np.linalg.pinv(prespective_transform)
        
        # gunakan 3 titik setelahnya u/ pixel-to-metric ratio dalam variable pts
        # warped_pt berisi transformasi 3 titik pada bird-view
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
        # initialize distance_w, dan distance_h yg masing2 berisi jarak 180 cm dalam satuan pixel
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        
        # draw 4 titik transformasi pada frame video prespective-view
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
        # Memproses deteksi dengan pre-trained model YOLO-v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        
        boxes = []
        confidences = []
        classIDs = []   
        rects = []
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                # deteksi (hanya) orang pada frame
                # YOLO menggunakan dataset COCO dimana index human dalam--
                # --dataset berada pada index 0
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                                    
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        # initialize bottom-center dari setiap bounding-box yg terdeteksi, dan--
        # initialize rects yg berisi bottom-center untuk digunakan pada generating ID's
        person_points, rects = utills.get_transformed_points(boxes1, prespective_transform)
        
        # initialize objects yg berisi ID's dan centroid dari object yg terdeteksi
        objects = ct.update(rects)
        
#         for (objectID, centroid) in objects.items():
#             text = "ID {}".format(objectID)
#             centroid = np.array([[[centroid[0], centroid[1]]]] , dtype="float32")
#             normal = cv2.perspectiveTransform(centroid, inv_trans)
#             cv2.putText(frame, text, (int(normal[0][0][0]) - 10, int(normal[0][0][1]) + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
# #             cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frame1 = np.copy(frame)
        
        # Hitung dan pelanggaran jarak antar objek (manusia) pada transformasi bird-view
        
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)
    
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor 
        # Hasilkan video output transformasi bird-view, dan--
        # gunakan hasil u/ digabungkan dgn video output prespective-view 
        bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, 
                                        risk_count, objects)
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count, bird_image)
        
        # resizing video output
        img = imutils.resize(img, height=720)
        
        # write video
        if count != 0:
            output_movie.write(img)
            #bird_movie.write(bird_image)
            
            #cv2.imshow('Bird Eye View', bird_image)
            #cv2.imwrite("frame%d.jpg" % count, img)
            #cv2.imwrite("Bird%d.jpg" % count, bird_image)
    
        count = count + 1


# In[ ]:


# initialize CentroidTracker Class
ct = CentroidTracker()

# initialize confidence dan threshold
confid = 0.5
thresh = 0.5

# initialize 7 titik transformasi prespective-view ke bird-view
# 4 titik pertama (bottom-left, bottom-right, top-right, top-left) digunakan u/--
# --melakukan transformasi prespective ke bird-view
# 3 titik setelahnya digunakan untuk menghitung pixel-to-metric ratio
pts = [(27.29, 559.83), (1408.92, 815.57), (1957.83, 30.41), (1182.81, 30.41), 
       (909.91, 642.47), (1057.27, 674.44), (1013.61, 559.83)]

# Load Yolov3 weights
weightsPath = "../input/yolo-coco-data/yolov3.weights"
configPath = "../input/yolo-coco-data/yolov3.cfg"

net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net_yl.getLayerNames()
ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

net_yl.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_yl.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# initialize video path
video_path = "../input/social-distancing/pedestrians.mp4"

output_dir = "../input/output/"
output_vid = "../input/output/"

# processing social distancing detector
calculate_social_distancing(video_path, net_yl, output_dir, output_vid, ln1, pts)

