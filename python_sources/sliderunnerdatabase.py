"""

    SlideRunner 

    https://www5.cs.fau.de/~aubreville/

   Database functions and annotation functions
"""

import sqlite3
import os
import time
import numpy as np
import random
import uuid


import cv2
import matplotlib.path as path
import numpy as np
class AnnotationType(enumerate):
    SPOT = 1
    AREA = 2
    POLYGON = 3
    SPECIAL_SPOT = 4
    CIRCLE = 5
    UNKNOWN = 255

class ViewingProfile(object):
    blindMode = False
    annotator = None
    COLORS_CLASSES = [[0,0,0,0],
                  [0,0,255,255],
                  [0,255,0,255],
                  [255,255,0,255],
                  [255,0,255,255],
                  [0,127,0,255],
                  [255,127,0,255],
                  [127,127,0,255],
                  [255,200,200,255],
                  [10, 166, 168,255],
                  [166, 10, 168,255],
                  [166,168,10,255]]
    spotCircleRadius = 25
    minimumAnnotationLabelZoom = 4
    majorityClassVote = False
    activeClasses = dict()



class AnnotationLabel(object):
    def __init__(self,annotatorId:int, classId:int, uid:int):
        self.annnotatorId = annotatorId
        self.classId = classId
        self.uid = uid
    

class annoCoordinate(object):
    x = None
    y = None
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def totuple(self):
        return (self.x, self.y)
    
    def tolist(self):
        return [self.x, self.y]

class AnnotationHandle(object):
    pt1 = None
    pt2 = None

    def __init__(self, pt1:annoCoordinate, pt2:annoCoordinate):
        self.pt1 = pt1
        self.pt2 = pt2

    def positionWithinRectangle(self,position:tuple):
        return ((position[0]>self.pt1.x) and (position[1]>self.pt1.y) and
                (position[0]<self.pt2.x) and (position[1]<self.pt2.y))



class annotation():

      def __init__(self, uid=0, text='',pluginAnnotationLabel=None):
          self.annotationType = AnnotationType.UNKNOWN
          self.labels = list()
          self.uid = uid
          self.text = text
          self.agreedClass = None
          self.clickable = True
          self.guid = ''
          self.deleted = 0
          self.lastModified = 0.
          self.minimumAnnotationLabelZoom = 1
          self.pluginAnnotationLabel = None

      def draw(self, image: np.ndarray, leftUpper: tuple, zoomLevel: float, thickness: int = 1, vp: ViewingProfile = ViewingProfile(), selected=False):
            return
        
      def setAgreedClass(self, agreedClass):
          self.agreedClass = agreedClass
    
      def positionInAnnotation(self, position: list) -> bool:
            return False

      def intersectingWithAnnotation(self, anno) -> bool:
          return self.convertToPath().intersects_path(anno.convertToPath())

      def convertToPath(self):
            return path.Path([])

      def getAnnotationsDescription(self, db) -> list:
           retval = list()
           if (self.pluginAnnotationLabel is None):
                for idx,label in enumerate(self.labels):
                    annotatorName = db.getAnnotatorByID(label.annnotatorId)
                    className = db.getClassByID(label.classId)
                    retval.append(['Anno %d' % (idx+1), '%s (%s)' % (className,annotatorName)])
                retval.append(['Agreed Class', db.getClassByID(self.agreedLabel())])
           else:
                retval.append(['Plugin class', str(self.pluginAnnotationLabel)])
                if (self.text is not None):
                    retval.append(['Description', str(self.text)])

           return retval
          
      def getBoundingBox(self) -> [int,int,int,int]:
        """
            returns the bounding box (x,y,w,h) for an object         
        """
        minC = self.minCoordinates()
        return [minC.x,minC.y] + list(self.getDimensions())

      def positionInAnnotationHandle(self, position: tuple) -> int:
          return None
    
          
          
      def getDescription(self, db, micronsPerPixel=None) -> list:
            if (self.pluginAnnotationLabel is None):
                return self.getAnnotationsDescription(db)
            else:
                return [['Plugin Anno',self.pluginAnnotationLabel.name],]

      def addLabel(self, label:AnnotationLabel, updateAgreed=True):
          self.labels.append(label)
          
          if (updateAgreed):
              self.agreedClass = self.majorityLabel()

      def _create_annohandle(self, image:np.ndarray, coord:tuple, markersize:int, color:tuple) -> AnnotationHandle:
            markersize=3
            pt1_rect = (max(0,coord[0]-markersize),
                        max(0,coord[1]-markersize))
            pt2_rect = (min(image.shape[1],coord[0]+markersize),
                        min(image.shape[0],coord[1]+markersize))
            cv2.rectangle(img=image, pt1=(pt1_rect), pt2=(pt2_rect), color=[255,255,255,255], thickness=2)
            cv2.rectangle(img=image, pt1=(pt1_rect), pt2=(pt2_rect), color=color, thickness=1)
            return AnnotationHandle(annoCoordinate(pt1_rect[0],pt1_rect[1]), annoCoordinate(pt2_rect[0],pt2_rect[1]))

      def getDimensions(self) -> (int, int):
          minC = self.minCoordinates()
          maxC = self.maxCoordinates()
          return (int(maxC.x-minC.x),int(maxC.y-minC.y))
        
      def getCenter(self) -> (annoCoordinate):
          minC = self.minCoordinates()
          maxC = self.maxCoordinates()
          return annoCoordinate(int(0.5*(minC.x+maxC.x)),int(0.5*(minC.y+maxC.y)))
        
      def removeLabel(self, uid:int):
          for label in range(len(self.labels)):
              if (self.labels[label].uid == uid):
                  self.labels.pop(label)
     
      def changeLabel(self, uid:int, annotatorId:int, classId:int):
          for label in range(len(self.labels)):
              if (self.labels[label].uid == uid):
                  self.labels[label] = AnnotationLabel(annotatorId, classId, uid)
      
      def maxLabelClass(self):
          retVal=0
          for label in range(len(self.labels)):
              if (self.labels[label].classId > retVal):
                  retVal=self.labels[label].classId
          return retVal


      """
            Returns the majority label for an annotation
      """
      def majorityLabel(self):
          if len(self.labels)==0:
              return 0

          histo = np.zeros(self.maxLabelClass()+1)

          for label in np.arange(0, len(self.labels)):
               histo[self.labels[label].classId] += 1

          if np.sum(histo == np.max(histo))>1:
              # no clear maximum, return 0
              return 0
          else:   
              # a clear winner. Return it.
              return np.argmax(histo)


      """
            Returns the agreed (common) label for an annotation
      """
      def agreedLabel(self):
          if (self.agreedClass is not None):
              return self.agreedClass
          else:
              return 0

      
      def labelBy(self, annotatorId):
          for label in np.arange(0, len(self.labels)):
               if (self.labels[label].annnotatorId == annotatorId):
                    return self.labels[label].classId
          return 0
        
      def getColor(self, vp : ViewingProfile):
          if (self.pluginAnnotationLabel is not None):
              return self.pluginAnnotationLabel.color
          if (vp.blindMode):
            return vp.COLORS_CLASSES[self.labelBy(vp.annotator) % len(vp.COLORS_CLASSES)]
          elif (vp.majorityClassVote):
            return vp.COLORS_CLASSES[self.majorityLabel() % len(vp.COLORS_CLASSES)]
          else:
            return vp.COLORS_CLASSES[self.agreedLabel() % len(vp.COLORS_CLASSES)]
      
      def minCoordinates(self) -> annoCoordinate:
            print('Whopsy... you need to overload this.')
            return annoCoordinate(None, None)

      def maxCoordinates(self) -> annoCoordinate:
            print('Whopsy... you need to overload this.')
            return annoCoordinate(None, None)


class rectangularAnnotation(annotation):
      def __init__(self, uid, x1, y1, x2, y2, text='', pluginAnnotationLabel=None, minimumAnnotationLabelZoom=1):
            super().__init__(uid=uid, text=text, pluginAnnotationLabel=pluginAnnotationLabel)
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.minimumAnnotationLabelZoom = minimumAnnotationLabelZoom
            self.annotationType = AnnotationType.AREA
      
      def minCoordinates(self) -> annoCoordinate:
            return annoCoordinate(self.x1, self.y1)
      
      def getDescription(self,db, micronsPerPixel=None) -> list:
          return [['Position', 'x1=%d, y1=%d, x2=%d, y2=%d' % (self.x1,self.y1,self.x2,self.y2)]] + self.getAnnotationsDescription(db)

      def maxCoordinates(self) -> annoCoordinate:
            return annoCoordinate(self.x2, self.y2)

      def positionInAnnotation(self, position: list) -> bool:
            return ((position[0]>self.x1) and (position[0]<self.x2) and 
                   (position[1]>self.y1) and (position[1]<self.y2))
    
      def width(self) -> float:
          return self.x2-self.x1

      def height(self) -> float:
          return self.y2-self.y1
        
      def centerCoordinate(self) -> tuple:
          return annoCoordinate(x=int(0.5*(self.x1+self.x2)),y=int(0.5*(self.y1+self.y2)))

      def convertToPath(self):
          return path.Path(np.array([[self.x1, self.x2, self.x2, self.x1, self.x1] ,[self.y1, self.y1, self.y2, self.y2, self.y1]]).T)

      def draw(self, image: np.ndarray, leftUpper: tuple, zoomLevel: float, thickness: int, vp : ViewingProfile, selected = False):
            xpos1=max(0,int((self.x1-leftUpper[0])/zoomLevel))
            ypos1=max(0,int((self.y1-leftUpper[1])/zoomLevel))
            xpos2=min(image.shape[1],int((self.x2-leftUpper[0])/zoomLevel))
            ypos2=min(image.shape[0],int((self.y2-leftUpper[1])/zoomLevel))

            image = cv2.rectangle(image, thickness=thickness, pt1=(xpos1,ypos1), pt2=(xpos2,ypos2),color=self.getColor(vp), lineType=cv2.LINE_AA)

            if (len(self.text)>0) and (zoomLevel < self.minimumAnnotationLabelZoom):
                  cv2.putText(image, self.text, (xpos1+3, ypos2+10), cv2.FONT_HERSHEY_PLAIN , 0.7,(0,0,0),1,cv2.LINE_AA)

class polygonAnnotation(annotation):
    def __init__(self, uid:int, coordinates: np.ndarray = None, text='', pluginAnnotationLabel=None, minimumAnnotationLabelZoom=1):
        super().__init__(uid=uid, pluginAnnotationLabel=pluginAnnotationLabel, text=text)
        self.annotationType = AnnotationType.POLYGON
        self.annoHandles = list()
        self.minimumAnnotationLabelZoom = minimumAnnotationLabelZoom
        if (coordinates is not None):
            self.coordinates = coordinates
    
    def positionInAnnotationHandle(self, position: tuple) -> int:
        for key,annoHandle in enumerate(self.annoHandles):
             if (annoHandle.positionWithinRectangle(position)):
                 return key
        return None

    def minCoordinates(self) -> annoCoordinate:
        return annoCoordinate(self.coordinates[:,0].min(), self.coordinates[:,1].min())

    def maxCoordinates(self) -> annoCoordinate:
        return annoCoordinate(self.coordinates[:,0].max(), self.coordinates[:,1].max())
    
    def area_px(self) -> float:
        return cv2.contourArea(self.coordinates)

    # Largest diameter --> 2* radius of minimum enclosing circle
    def diameter_px(self) -> float:
        (x,y),radius = cv2.minEnclosingCircle(self.coordinates)
        return radius*2

    def getDescription(self,db, micronsPerPixel=None) -> list:
        mc = annoCoordinate(self.coordinates[:,0].mean(), self.coordinates[:,1].mean())
        area_px = float(self.area_px())
        diameter_px = float(self.diameter_px())
        micronsPerPixel = float(micronsPerPixel)
        if micronsPerPixel < 2E-6:
            area = '%d px^2' % area_px
            diameter = '%d px^2' % diameter_px
        else:
            area_mum2 = (area_px*micronsPerPixel*micronsPerPixel)
            diameter_mum = diameter_px*micronsPerPixel
            if (area_mum2 < 1E4):
                area = '%.2f µm^2' % area_mum2
            else:
                area = '%.2f mm^2' % (1E-6 * area_mum2)
            if (diameter_mum < 1e3):
                diameter = '%.2f µm^2' % diameter_mum
            else:
                diameter = '%.2f mm^2' % (diameter_mum * 1e-3)

        return [['Position', 'x1=%d, y1=%d' % (mc.x,mc.y)], ['Area', area], ['Largest diameter', diameter]] + self.getAnnotationsDescription(db)

    def convertToPath(self):
        p = path.Path(self.coordinates)
        return p

    def positionInAnnotation(self, position: list) -> bool:
        return self.convertToPath().contains_point(position)

    def draw(self, image: np.ndarray, leftUpper: tuple, zoomLevel: float, thickness: int, vp : ViewingProfile, selected=False):
        def slideToScreen(pos):
            """
                convert slide coordinates to screen coordinates
            """
            xpos,ypos = pos
            p1 = leftUpper
            cx = int((xpos - p1[0]) / zoomLevel)
            cy = int((ypos - p1[1]) / zoomLevel)
            return (cx,cy)        
        markersize = min(3,int(5/zoomLevel))
        listIdx=-1

        self.annoHandles=list()

        # small assertion to fix bug #12
        if (self.coordinates.shape[1]==0):
            return image

        for listIdx in range(self.coordinates.shape[0]-1):
            anno = slideToScreen(self.coordinates[listIdx])
            cv2.line(img=image, pt1=anno, pt2=slideToScreen(self.coordinates[listIdx+1]), thickness=2, color=self.getColor(vp), lineType=cv2.LINE_AA)       

            if (selected):
                self.annoHandles.append(self._create_annohandle(image, anno, markersize, self.getColor(vp)))


        listIdx+=1
        anno = slideToScreen(self.coordinates[listIdx])
        if (selected):
                self.annoHandles.append(self._create_annohandle(image, anno, markersize, self.getColor(vp)))

        cv2.line(img=image, pt1=anno, pt2=slideToScreen(self.coordinates[0]), thickness=2, color=self.getColor(vp), lineType=cv2.LINE_AA)       

        if (len(self.text)>0) and (zoomLevel < self.minimumAnnotationLabelZoom):
                xpos1=int(0.5*(np.max(self.coordinates[:,0])+np.min(self.coordinates[:,0]) ))
                ypos1=int(0.5*(np.max(self.coordinates[:,1])+np.min(self.coordinates[:,1])))
                cv2.putText(image, self.text, slideToScreen((xpos1+3, ypos1+10)), cv2.FONT_HERSHEY_PLAIN , 0.7,(0,0,0),1,cv2.LINE_AA)



class circleAnnotation(annotation):
      
      def __init__(self, uid, x1, y1, x2 = None, y2 = None, r = None, text='', pluginAnnotationLabel=None, minimumAnnotationLabelZoom=1):
            super().__init__(uid=uid, text=text, pluginAnnotationLabel=pluginAnnotationLabel)
            self.annotationType = AnnotationType.CIRCLE
            self.minimumAnnotationLabelZoom = minimumAnnotationLabelZoom

            if (r is None):
                self.x1 = int(0.5*(x1+x2))
                self.y1 = int(0.5*(y1+y2))
                if (x2>x1):
                    self.r = int((x2-x1)*0.5)
                else:
                    self.r = int((x1-x2)*0.5)
            else:
                self.x1 = int(x1)
                self.y1 = int(y1)
                self.r = int(r)

      def minCoordinates(self) -> annoCoordinate:
            return annoCoordinate(self.x1-self.r, self.y1-self.r)

      def convertToPath(self):
          pi = np.linspace(0,2*np.pi,100)
          x = np.sin(pi)*self.r+self.x1
          y = np.cos(pi)*self.r+self.y1
          return path.Path(np.c_[x,y])
          

      def maxCoordinates(self) -> annoCoordinate:
            return annoCoordinate(self.x1+self.r, self.y1+self.r)

      def getDescription(self,db, micronsPerPixel=None) -> list:
          return [['Position', 'x1=%d, y1=%d' % (self.x1, self.y1)]] + self.getAnnotationsDescription(db)

      def positionInAnnotation(self, position: list) -> bool:
          dist = np.sqrt(np.square(position[0]-self.x1)+np.square(position[1]-self.y1))
          return (dist<=self.r)

      def draw(self, image: np.ndarray, leftUpper: tuple, zoomLevel: float, thickness: int, vp : ViewingProfile, selected=False):
            xpos1=int((self.x1-leftUpper[0])/zoomLevel)
            ypos1=int((self.y1-leftUpper[1])/zoomLevel)
            radius = int(self.r/zoomLevel)
            if (radius>=0):
                  image = cv2.circle(image, thickness=thickness, center=(xpos1,ypos1), radius=radius,color=self.getColor(vp), lineType=cv2.LINE_AA)
            if (len(self.text)>0) and (zoomLevel < self.minimumAnnotationLabelZoom):
                    cv2.putText(image, self.text, (xpos1,ypos1), cv2.FONT_HERSHEY_PLAIN , 0.7,(0,0,0),1,cv2.LINE_AA)

class spotAnnotation(annotation):

      def __init__(self, uid, x1, y1, isSpecialSpot : bool = False,text='', pluginAnnotationLabel=None, minimumAnnotationLabelZoom=1):
            super().__init__(uid=uid, text=text, pluginAnnotationLabel=pluginAnnotationLabel)
            self.annotationType = AnnotationType.SPOT
            self.x1 = x1
            self.y1 = y1
            if (isSpecialSpot):
                self.annotationType = AnnotationType.SPECIAL_SPOT
            self.minimumAnnotationLabelZoom = minimumAnnotationLabelZoom

      def minCoordinates(self) -> annoCoordinate:
            return annoCoordinate(self.x1-25, self.y1-25)

      def maxCoordinates(self) -> annoCoordinate:
            return annoCoordinate(self.x1+25, self.y1+25)

      def intersectingWithAnnotation(self, anno) -> bool:
            return False


      def draw(self, image: np.ndarray, leftUpper: tuple, zoomLevel: float, thickness: int, vp : ViewingProfile, selected=False):
            xpos1=int((self.x1-leftUpper[0])/zoomLevel)
            ypos1=int((self.y1-leftUpper[1])/zoomLevel)
            radius=int(int(vp.spotCircleRadius)/zoomLevel)
            if (radius>=0):
                  image = cv2.circle(image, thickness=thickness, center=(xpos1,ypos1), radius=radius,color=self.getColor(vp), lineType=cv2.LINE_AA)
            if (len(self.text)>0) and (zoomLevel < self.minimumAnnotationLabelZoom):
                    cv2.putText(image, self.text, (xpos1+3, ypos1+10), cv2.FONT_HERSHEY_PLAIN , 0.7,(0,0,0),1,cv2.LINE_AA)

      def getDescription(self,db, micronsPerPixel=None) -> list:
          return [['Position', 'x1=%d, y1=%d' % (self.x1, self.y1)]] + self.getAnnotationsDescription(db)

      def positionInAnnotation(self, position: list) -> bool:
          dist = np.sqrt(np.square(position[0]-self.x1)+np.square(position[1]-self.y1))
          return (dist<=25)

def generate_uuid():
    return str(uuid.uuid4())


class DatabaseField(object):
    def __init__(self, keyStr:str, typeStr:str, isNull:int=0, isUnique:bool=False, isAutoincrement:bool=False, defaultValue:str='', primaryKey:int=0):
        self.key = keyStr
        self.type = typeStr
        self.isNull = isNull
        self.isUnique = isUnique
        self.isAutoincrement = isAutoincrement
        self.defaultValue = defaultValue
        self.isPrimaryKey = primaryKey

    def creationString(self):
        return ("`"+self.key+"` "+self.type+
                ((" DEFAULT %s " % self.defaultValue) if not(self.defaultValue == "") else "") + 
                (" PRIMARY KEY" if self.isPrimaryKey else "")+
                (" AUTOINCREMENT" if self.isAutoincrement else "")+
                (" UNIQUE" if self.isUnique else "") )


def isActiveClass(label, activeClasses):
        if (label) is None:
            return True
        if (label<len(activeClasses)):
            return activeClasses[label]
        else:
            print('Warning: Assigned label is: ',label,' while class list is: ',activeClasses)
            return 0

class DatabaseTable(object):
    def __init__(self, name:str):
        self.entries = dict()
        self.name = name

    def add(self, field:DatabaseField):
        self.entries[field.key] = field
        return self

    def getCreateStatement(self):
        createStatement = "CREATE TABLE `%s` (" % self.name
        cnt=0
        for idx, entry in self.entries.items():
            if cnt>0:
                createStatement += ','
            createStatement += entry.creationString()
            cnt+=1
        createStatement += ");"
        return createStatement

    def checkTableInfo(self, tableInfo):
        if (tableInfo is None) or (len(tableInfo)==0):
            return False
        allKeys=list()
        for entry in tableInfo:
            idx,key,typeStr,isNull,defaultValue,PK = entry
            if (key not in self.entries):
                return False
            if not (self.entries[key].type == typeStr):
                return False
            allKeys += [key]
        for entry in self.entries:
            if entry not in allKeys:
                return False
        # OK, defaultValue, isNull are not important to be checked

        return True


    def addMissingTableInfo(self, tableInfo, tableName):
        returnStr = list()
        if (tableInfo is None) or (len(tableInfo)==0):
            return False
        allKeys=list()
        for entry in tableInfo:
            idx,key,typeStr,isNull,defaultValue,PK = entry
            allKeys += [key]

        for entry in self.entries.keys():
            if entry not in allKeys:
                returnStr.append('ALTER TABLE %s ADD COLUMN %s' % (tableName, self.entries[entry].creationString()))

        return returnStr


from typing import Dict

class Database(object):
    annotations = Dict[int,annotation]

    minCoords = np.empty(0)
    maxCoords = np.empty(0)

    def __init__(self):
        self.dbOpened = False
        self.VA = dict()

        self.databaseStructure = dict()
        self.transformer = None
        self.annotations = dict()       
        self.doCommit = True
        self.annotationsSlide = None
        self.databaseStructure['Log'] = DatabaseTable('Log').add(DatabaseField('uid','INTEGER',isAutoincrement=True, primaryKey=1)).add(DatabaseField('dateTime','FLOAT')).add(DatabaseField('labelId','INTEGER'))
        self.databaseStructure['Slides'] = DatabaseTable('Slides').add(DatabaseField('uid','INTEGER',isAutoincrement=True, primaryKey=1)).add(DatabaseField('filename','TEXT')).add(DatabaseField('width','INTEGER')).add(DatabaseField('height','INTEGER')).add(DatabaseField('directory','TEXT')).add(DatabaseField('uuid','TEXT'))
        self.databaseStructure['Annotations'] = DatabaseTable('Annotations').add(DatabaseField('uid','INTEGER',isAutoincrement=True, primaryKey=1)).add(DatabaseField('guid','TEXT')).add(DatabaseField('deleted','INTEGER',defaultValue=0)).add(DatabaseField('slide','INTEGER')).add(DatabaseField('type','INTEGER')).add(DatabaseField('agreedClass','INTEGER')).add(DatabaseField('lastModified','REAL',defaultValue=str(time.time())))

    def isOpen(self):
        return self.dbOpened

    def appendToMinMaxCoordsList(self, anno: annotation):
        self.minCoords = np.vstack((self.minCoords, np.asarray(anno.minCoordinates().tolist()+[anno.uid])))
        self.maxCoords = np.vstack((self.maxCoords, np.asarray(anno.maxCoordinates().tolist()+[anno.uid])))
    def generateMinMaxCoordsList(self):
        # MinMaxCoords lists shows extreme coordinates from object, to decide if an object shall be shown
        self.minCoords = np.zeros(shape=(len(self.annotations),3))
        self.maxCoords = np.zeros(shape=(len(self.annotations),3))
        keys = self.annotations.keys() 
        for idx,annokey in enumerate(keys):
            annotation = self.annotations[annokey]
            self.minCoords[idx] = np.asarray(annotation.minCoordinates().tolist()+[annokey])
            self.maxCoords[idx] = np.asarray(annotation.maxCoordinates().tolist()+[annokey])

#        print(self.getVisibleAnnotations([0,0],[20,20]))
    
    def getVisibleAnnotations(self, leftUpper:list, rightLower:list) -> Dict[int, annotation]:
        potentiallyVisible =  ( (self.maxCoords[:,0] > leftUpper[0]) & (self.minCoords[:,0] < rightLower[0]) & 
                                (self.maxCoords[:,1] > leftUpper[1]) & (self.minCoords[:,1] < rightLower[1]) )
        ids = self.maxCoords[potentiallyVisible,2]
        return dict(filter(lambda i:i[0] in ids, self.annotations.items()))

    def listOfSlides(self):
        self.execute('SELECT uid,filename from Slides')
        return self.fetchall()


    def annotateImage(self, img: np.ndarray, leftUpper: list, rightLower:list, zoomLevel:float, vp : ViewingProfile, selectedAnnoID:int):
        annos = self.getVisibleAnnotations(leftUpper, rightLower)
        self.VA = annos
        for idx,anno in annos.items():
            if (isActiveClass(activeClasses=vp.activeClasses,label=anno.agreedLabel())) and not anno.deleted:
                anno.draw(img, leftUpper, zoomLevel, thickness=2, vp=vp, selected=(selectedAnnoID==anno.uid))
    
    def findIntersectingAnnotation(self, anno:annotation, vp: ViewingProfile, database=None, annoType = None):    
        if (database is None):
            database = self.VA            
        for idx,DBanno in database.items():
            if (vp.activeClasses[DBanno.agreedLabel()]):
                if (DBanno.intersectingWithAnnotation(anno )):
                    if (annoType == DBanno.annotationType) or (annoType is None):
                        return DBanno
        return None

    def findClickAnnotation(self, clickPosition, vp : ViewingProfile, database=None, annoType = None):
        if (database is None):
            database = self.VA            
        for idx,anno in database.items():
            if (vp.activeClasses[anno.agreedLabel()]):
                if (anno.positionInAnnotation(clickPosition )) and (anno.clickable):
                    if (annoType == anno.annotationType) or (annoType is None):
                        return anno
        return None

    def loadIntoMemory(self, slideId, transformer=None):
        self.annotations = dict()
        self.annotationsSlide = slideId
        self.guids = dict()
        self.transformer = transformer

        if (slideId is None):
            return

        self.dbcur.execute('SELECT uid, type,agreedClass,guid,lastModified,deleted FROM Annotations WHERE slide == %d'% slideId)
        allAnnos = self.dbcur.fetchall()


        self.dbcur.execute('SELECT coordinateX, coordinateY,annoid FROM Annotations_coordinates where annoId IN (SELECT uid FROM Annotations WHERE slide == %d) ORDER BY orderIdx' % (slideId))
        allCoords = np.asarray(self.dbcur.fetchall())

        if self.transformer is not None:
            allCoords = self.transformer(allCoords)

        for uid, annotype,agreedClass,guid,lastModified,deleted in allAnnos:
            coords = allCoords[allCoords[:,2]==uid,0:2]
            if (annotype == AnnotationType.SPOT):
                self.annotations[uid] = spotAnnotation(uid, coords[0][0], coords[0][1])
            elif (annotype == AnnotationType.SPECIAL_SPOT):
                self.annotations[uid] = spotAnnotation(uid, coords[0][0], coords[0][1], True)
            elif (annotype == AnnotationType.POLYGON):
                self.annotations[uid] = polygonAnnotation(uid, coords)
            elif (annotype == AnnotationType.AREA):
                self.annotations[uid] = rectangularAnnotation(uid, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
            elif (annotype == AnnotationType.CIRCLE):
                self.annotations[uid] = circleAnnotation(uid, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
            else:
                print('Unknown annotation type %d found :( ' % annotype)
            self.annotations[uid].agreedClass = agreedClass
            self.annotations[uid].guid = guid
            self.annotations[uid].lastModified = lastModified
            self.annotations[uid].deleted = deleted
            self.guids[guid] = uid
        # Add all labels
        self.dbcur.execute('SELECT annoid, person, class,uid FROM Annotations_label WHERE annoID in (SELECT uid FROM Annotations WHERE slide == %d)'% slideId)
        allLabels = self.dbcur.fetchall()

        for (annoId, person, classId,uid) in allLabels:
            self.annotations[annoId].addLabel(AnnotationLabel(person, classId, uid), updateAgreed=False)


        self.generateMinMaxCoordsList()

            
    

    def checkTableStructure(self, tableName, action='check'):
        self.dbcur.execute('PRAGMA table_info(%s)' % tableName)
        ti = self.dbcur.fetchall()
        if (action=='check'):
            return self.databaseStructure[tableName].checkTableInfo(ti)
        elif (action=='ammend'):
            return self.databaseStructure[tableName].addMissingTableInfo(ti, tableName)

    def open(self, dbfilename):
        if os.path.isfile(dbfilename):
            self.db = sqlite3.connect(dbfilename)
            self.dbcur = self.db.cursor()
            self.db.create_function("generate_uuid",0, generate_uuid)
            self.db.create_function("pycurrent_time",0, time.time)

            # Check structure of database and ammend if not proper
            if not self.checkTableStructure('Slides'):
                sql_statements = self.checkTableStructure('Slides','ammend')
                for sql in sql_statements:
                    self.dbcur.execute(sql)
                self.db.commit()

            if not self.checkTableStructure('Annotations'):
                sql_statements = self.checkTableStructure('Annotations','ammend')
                for sql in sql_statements:
                    self.dbcur.execute(sql)
                self.db.commit()

            if not self.checkTableStructure('Log'):
                # add new log, no problemo.
                self.dbcur.execute('DROP TABLE if exists `Log`')
                self.dbcur.execute(self.databaseStructure['Log'].getCreateStatement())
                self.db.commit()
            
            # Migrating vom V0 to V1 needs proper filling of GUIDs
            DBversion = self.dbcur.execute('PRAGMA user_version').fetchone()
            if (DBversion[0]==0):
                self.dbcur.execute('UPDATE Annotations set guid=generate_uuid() where guid is NULL')

                self.addTriggers()

                # Add last polygon point (close the line)
                allpolys = self.dbcur.execute('SELECT uid from Annotations where type==3').fetchall()
                for [polyid,] in allpolys:
                    coords = self.dbcur.execute(f'SELECT coordinateX, coordinateY,slide FROM Annotations_coordinates where annoid=={polyid} and orderidx==1').fetchone()
                    maxidx = self.dbcur.execute(f'SELECT MAX(orderidx) FROM Annotations_coordinates where annoid=={polyid}').fetchone()[0]+1
                    self.dbcur.execute(f'INSERT INTO Annotations_coordinates (coordinateX, coordinateY, slide, annoId, orderIdx) VALUES ({coords[0]},{coords[1]},{coords[2]},{polyid},{maxidx} )')

                DBversion = self.dbcur.execute('PRAGMA user_version = 1')
                print('Successfully migrated DB to version 1')
                self.commit()

            self.dbOpened = True
            self.dbfilename = dbfilename
            self.dbname = os.path.basename(dbfilename)



            return self
        else:
            return False
    
    def deleteTriggers(self):
        for event in ['UPDATE','INSERT']:
            self.dbcur.execute(f'DROP TRIGGER IF EXISTS updateAnnotation_fromLabel{event}')
            self.dbcur.execute(f'DROP TRIGGER IF EXISTS updateAnnotation_fromCoords{event}')
            self.dbcur.execute(f'DROP TRIGGER IF EXISTS updateAnnotation_{event}')

    def addTriggers(self):
        for event in ['UPDATE','INSERT']:
            self.dbcur.execute(f"""CREATE TRIGGER IF NOT EXISTS updateAnnotation_fromLabel{event}
                        AFTER {event} ON Annotations_label
                        BEGIN
                            UPDATE Annotations SET lastModified=pycurrent_time() where uid==new.annoId;
                        END
                        ;
                        """)

            self.dbcur.execute(f"""CREATE TRIGGER IF NOT EXISTS updateAnnotation_fromCoords{event}
                        AFTER {event} ON Annotations_coordinates
                        BEGIN
                            UPDATE Annotations SET lastModified=pycurrent_time() where uid==new.annoId;
                        END
                        ;
                        """)
            self.dbcur.execute(f"""CREATE TRIGGER IF NOT EXISTS updateAnnotation_{event}
                        AFTER {event} ON Annotations
                        BEGIN
                            UPDATE Annotations SET lastModified=pycurrent_time() where uid==new.uid;
                        END
                        ;
                        """)
    
    # copy database to new file
    def saveTo(self, dbfilename):
        new_db = sqlite3.connect(dbfilename) # create a memory database

        query = "".join(line for line in self.db.iterdump())

        # Dump old database in the new one. 
        new_db.executescript(query)

        return True

    def findSpotAnnotations(self,leftUpper, rightLower, slideUID, blinded = False, currentAnnotator=None):
        q = ('SELECT coordinateX, coordinateY, agreedClass,Annotations_coordinates.uid,annoId,type FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND (type==1 OR type==4)'%(slideUID) )
        if not blinded:
            self.execute(q)
            return self.fetchall()
    
        q=(' SELECT coordinateX, coordinateY,0,uid,annoId,1 from Annotations_coordinates WHERE coordinateX >= '+str(leftUpper[0])+
                 ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+' AND slide == %d;' % slideUID)
                
        self.execute(q)
        resp = np.asarray(self.fetchall())

        if (resp.shape[0]==0):
            return list()


        self.execute('SELECT annoId, class from Annotations_label LEFT JOIN Annotations on Annotations_label.annoId == Annotations.uid WHERE person==%d and TYPE IN (1,4) GROUP BY annoId' % currentAnnotator)
        myAnnos = np.asarray(self.fetchall())

        if (myAnnos.shape[0]==0):
            myOnes = np.zeros(resp.shape).astype(np.bool)
            mineInAll = np.empty(0)
        else:
            myOnes = np.in1d(resp[:,4],myAnnos[:,0])
            mineInAll = np.in1d(myAnnos[:,0],resp[:,4])

        if mineInAll.shape[0]>0:
            resp[myOnes,2] = myAnnos[mineInAll,1]

        self.execute('SELECT uid,type from Annotations WHERE type IN (1,4) AND slide == %d'% slideUID)
        correctTypeUIDs = np.asarray(self.fetchall())
        if (correctTypeUIDs.shape[0]==0):
            return list() # No annotation with correct type available

        typeFilter = np.in1d(resp[:,4], correctTypeUIDs[:,0])
        assignType = np.in1d(correctTypeUIDs[:,0], resp[:,4])
        resp[:,5] = correctTypeUIDs[assignType,1]
        return resp[typeFilter,:].tolist()
 

    def getUnknownInCurrentScreen(self,leftUpper, rightLower, currentAnnotator) -> annotation:
        visAnnos = self.getVisibleAnnotations(leftUpper, rightLower)
        for anno in visAnnos.keys():
            if (visAnnos[anno].labelBy(currentAnnotator) == 0):
                return anno   
        return None

    def findAllAnnotationLabels(self, uid):
        q = 'SELECT person, class, uid FROM Annotations_label WHERE annoId== %d' % uid
        self.execute(q)
        return self.fetchall()

    def checkIfAnnotatorLabeled(self, uid, person):
        q = 'SELECT COUNT(*) FROM Annotations_label WHERE annoId== %d AND person == %d' % (uid, person)
        self.execute(q)
        return self.fetchone()[0]

    def findClassidOfClass(self, classname):
        q = 'SELECT uid FROM Classes WHERE name == "%s"' % classname
        self.execute(q)
        return self.fetchall()

    def findAllAnnotations(self, annoId, slideUID = None):
        if (slideUID is None):
            q = 'SELECT coordinateX, coordinateY FROM Annotations_coordinates WHERE annoId==%d' % annoId
        else:
            q = 'SELECT coordinateX, coordinateY FROM Annotations_coordinates WHERE annoId==%d AND slide == %d' % (annoId, slideUID)
        self.execute(q)
        return self.fetchall()    

    def findSlideForAnnotation(self, annoId):
        q = 'SELECT filename FROM Annotations_coordinates LEFT JOIN Slides on Slides.uid == Annotations_coordinates.slide WHERE annoId==%d' % annoId
        self.execute(q)
        return self.fetchall()    


    def pickRandomUnlabeled(self, byAnnotator=0) -> annotation:
        annoIds = list(self.annotations.keys())
        random.shuffle(annoIds)
        for annoId in annoIds:
            if (self.annotations[annoId].labelBy(byAnnotator) == 0):
                return self.annotations[annoId]
        return None

    def findPolygonAnnotatinos(self,leftUpper,rightLower, slideUID,blinded = False, currentAnnotator=None):
        if not blinded:
            q = ('SELECT agreedClass,annoId FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND type==3 '%(slideUID) +' GROUP BY Annotations_coordinates.annoId')
            self.execute(q)
            farr = self.fetchall()
        else:
            q = ('SELECT 0,annoId FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND type==3 '%(slideUID) +' AND Annotations.uid NOT IN (SELECT annoId FROM Annotations_label WHERE person==%d GROUP BY annoID) GROUP BY Annotations_coordinates.annoId' % currentAnnotator)
            self.execute(q)
            farr1 = self.fetchall()

            q = ('SELECT class,Annotations_label.annoId FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId LEFT JOIN Annotations_label ON Annotations_label.annoId == Annotations.uid WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND type==3 '%(slideUID) +' AND Annotations_label.person == %d GROUP BY Annotations_coordinates.annoId' % currentAnnotator)
            self.execute(q)
            farr2 = self.fetchall()

            farr = farr1+farr2


        polysets = []
        
        toggler=True
        for entryOuter in range(len(farr)):
            # find all annotations for area:
            allAnnos = self.findAllAnnotations(farr[entryOuter][1])
            polygon = list()
            for entry in np.arange(0,len(allAnnos),1):
                polygon.append([allAnnos[entry][0],allAnnos[entry][1]])
            polygon.append([allAnnos[0][0],allAnnos[0][1]]) # close polygon
            polysets.append((polygon,farr[entryOuter][0],farr[entryOuter][1] ))

        return polysets

    def findAreaAnnotations(self,leftUpper, rightLower, slideUID, blinded = False, currentAnnotator = 0):
        if not blinded:
            q = ('SELECT coordinateX, coordinateY,agreedClass, annoId, type, orderIdx FROM Annotations_coordinates  LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE annoID in (SELECT annoId FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND type IN (2,5) '%(slideUID) +' group by Annotations_coordinates.annoId) ORDER BY Annotations_coordinates.annoId, orderIdx')
            self.execute(q)
            farr = self.fetchall()

            reply = []
            for entry in range(len(farr)-1):
                # find all annotations for area:
                if (farr[entry][5]==1):
                    reply.append([farr[entry][0],farr[entry][1],farr[entry+1][0],farr[entry+1][1],farr[entry][2],farr[entry][3],farr[entry][4]])
                    # tuple: x1,y1,x2,y2,class,annoId ID, type
            return reply

        else:
            q = ('SELECT 0,annoId,type FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND type IN (2,5) '%(slideUID) +' AND Annotations.uid NOT IN (SELECT annoId FROM Annotations_label WHERE person==%d GROUP BY annoID) GROUP BY Annotations_coordinates.annoId' % currentAnnotator)

            self.execute(q)
            farr1 = self.fetchall()

            q = ('SELECT class,Annotations_label.annoId,type FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId LEFT JOIN Annotations_label ON Annotations_label.annoId == Annotations.uid WHERE coordinateX >= '+str(leftUpper[0])+
                ' AND coordinateX <= '+str(rightLower[0])+' AND coordinateY >= '+str(leftUpper[1])+' AND coordinateY <= '+str(rightLower[1])+
                ' AND Annotations.slide == %d AND type IN (2,5) '%(slideUID) +' AND Annotations_label.person == %d GROUP BY Annotations_coordinates.annoId' % currentAnnotator)

            self.execute(q)
            farr2 = self.fetchall()

            farr = farr1+farr2
            
        reply = []
        toggler=True
        for entryOuter in range(len(farr)):
            # find all annotations for area:
            allAnnos = self.findAllAnnotations(farr[entryOuter][1])
            for entry in np.arange(0,len(allAnnos),2):
                reply.append([allAnnos[entry][0],allAnnos[entry][1],allAnnos[entry+1][0],allAnnos[entry+1][1],farr[entryOuter][0],farr[entryOuter][1],farr[entryOuter][2]])
            # tuple: x1,y1,x2,y2,class,annoId ID, type
        return reply


    def setSlideDimensions(self,slideuid,dimensions):
        if dimensions is None:
            return
        if (slideuid is None):
            return
        if (dimensions[0] is None):
            return
        if (dimensions[1] is None):
            return
        print('Setting dimensions of slide ',slideuid,'to',dimensions)
        self.execute('UPDATE Slides set width=%d, height=%d WHERE uid=%d' % (dimensions[0],dimensions[1],slideuid))
        self.db.commit()

    def findSlideWithFilename(self,slidename,slidepath, uuid:str=None):
        if (len(slidepath.split(os.sep))>1):
            directory = slidepath.split(os.sep)[-2]
        else:
            directory = ''
        ret = self.execute('SELECT uid,directory,uuid,filename from Slides ').fetchall()
        secondBest=None
        for (uid,slidedir,suuid,fname) in ret:
            if (uuid is not None) and (suuid==uuid):
                return uid
            elif (fname==slidename):
                if slidedir is None:
                    secondBest=uid
                elif (slidedir.upper() == directory.upper()):
                    return uid
                else:
                    secondBest=uid
        return secondBest
    
    def insertAnnotator(self, name):
        self.execute('INSERT INTO Persons (name) VALUES ("%s")' % (name))
        self.commit()
        query = 'SELECT last_insert_rowid()'
        return self.execute(query).fetchone()[0]

    def insertClass(self, name):
        self.execute('INSERT INTO Classes (name) VALUES ("%s")' % (name))
        self.commit()
    
    def setAgreedClass(self, classId, annoIdx):
        self.annotations[annoIdx].agreedClass = classId
        q = 'UPDATE Annotations SET agreedClass==%d WHERE uid== %d' % (classId,annoIdx)
        self.execute(q)
        self.commit()
         

    def setAnnotationLabel(self,classId,  person, entryId, annoIdx):
        q = 'UPDATE Annotations_label SET person==%d, class=%d WHERE uid== %d' % (person,classId,entryId)
        self.execute(q)
        self.commit()
        self.annotations[annoIdx].changeLabel(entryId, person, classId)
        
        # check if all labels belong to one class now
        if np.all(np.array([lab.classId for lab in self.annotations[annoIdx].labels])==classId):
            self.setAgreedClass(classId, annoIdx)

        self.checkCommonAnnotation(annoIdx)

    def checkCommonAnnotation(self, annoIdx ):
        allAnnos = self.findAllAnnotationLabels(annoIdx)
        if (len(allAnnos)>0):
            matching = allAnnos[0][1]
            for anno in allAnnos:
                if (anno[1] != matching):
                    matching=0
        else:
            matching=0
        
        q = 'UPDATE Annotations set agreedClass=%d WHERE UID=%d' % (matching, annoIdx)
        self.execute(q)
        self.commit()

    def logLabel(self, labelId):
        query = 'INSERT INTO Log (dateTime, labelId) VALUES (%d, %d)' % (time.time(), labelId)
        self.execute(query)


    def addAnnotationLabel(self,classId,  person, annoId):
        query = ('INSERT INTO Annotations_label (person, class, annoId) VALUES (%d,%d,%d)'
                 % (person, classId, annoId))
        self.execute(query)
        query = 'SELECT last_insert_rowid()'
        self.execute(query)
        newid = self.fetchone()[0]
        self.logLabel(newid)
        self.annotations[annoId].addLabel(AnnotationLabel(person, classId, newid))
        self.checkCommonAnnotation( annoId)
        self.commit()

    def exchangePolygonCoordinates(self, annoId, slideUID, annoList):
        self.annotations[annoId].annotationType = AnnotationType.POLYGON
        self.annotations[annoId].coordinates = np.asarray(annoList)
        self.generateMinMaxCoordsList()

        query = 'DELETE FROM Annotations_coordinates where annoId == %d' % annoId
        self.execute(query)
        self.commit()

        self.insertCoordinates(np.array(annoList), slideUID, annoId)
        

    def insertCoordinates(self, annoList:np.ndarray, slideUID, annoId):
        """
                Insert an annotation into the database.
                annoList must be a numpy array, but can be either 2 columns or 3 columns (3rd is order)
        """

        if self.transformer is not None:
            annoList = self.transformer(annoList, inverse=True)

        for num, annotation in enumerate(annoList.tolist()):
            query = ('INSERT INTO Annotations_coordinates (coordinateX, coordinateY, slide, annoId, orderIdx) VALUES (%d,%d,%d,%d,%d)'
                    % (annotation[0],annotation[1],slideUID, annoId, annotation[2] if len(annotation)>2 else num+1))
            self.execute(query)
        self.commit()


    def insertNewPolygonAnnotation(self, annoList, slideUID, classID, annotator, closed:bool=True):
        query = 'INSERT INTO Annotations (slide, agreedClass, type) VALUES (%d,%d,3)' % (slideUID,classID)
#        query = 'INSERT INTO Annotations (coordinateX1, coordinateY1, coordinateX2, coordinateY2, slide, class1, person1) VALUES (%d,%d,%d,%d,%d,%d, %d)' % (x1,y1,x2,y2,slideUID,classID,annotator)
        if (isinstance(annoList, np.ndarray)):
            annoList=annoList.tolist()
        if (closed):
            annoList.append(annoList[0])
        self.execute(query)
        query = 'SELECT last_insert_rowid()'
        self.execute(query)
        annoId = self.fetchone()[0]
        assert(len(annoList)>0)
        self.annotations[annoId] = polygonAnnotation(annoId, np.asarray(annoList))
        self.appendToMinMaxCoordsList(self.annotations[annoId])
        self.insertCoordinates(np.array(annoList), slideUID, annoId)

        self.addAnnotationLabel(classId=classID, person=annotator, annoId=annoId)

        self.commit()
        return annoId

    def addAnnotationToDatabase(self, anno:annotation, slideUID:int, classID:int, annotatorID:int):
        if (anno.annotationType == AnnotationType.AREA):
            self.insertNewAreaAnnotation(anno.x1,anno.y1,anno.x2,anno.y2,slideUID,classID, annotatorID)
        elif (anno.annotationType == AnnotationType.POLYGON):
            self.insertNewPolygonAnnotation(anno.coordinates, slideUID, classID, annotatorID)
        elif (anno.annotationType == AnnotationType.CIRCLE):
            self.insertNewAreaAnnotation(anno.x1,anno.y1,anno.x2,anno.y2,slideUID,classID, annotatorID, typeId=5)
        elif (anno.annotationType == AnnotationType.SPOT):
            self.insertNewSpotAnnotation(anno.x1, anno.y1, slideUID, classID, annotatorID)
        elif (anno.annotationType == AnnotationType.SPECIAL_SPOT):
            self.insertNewSpotAnnotation(anno.x1, anno.y1, slideUID, classID, annotatorID, type=4)
        

    def getGUID(self, annoid) -> str:
        try:
            return self.execute(f'SELECT guid from Annotations where uid=={annoid}').fetchone()[0]
        except:
            return None

    def insertNewAreaAnnotation(self, x1,y1,x2,y2, slideUID, classID, annotator, typeId=2, uuid=None):
        query = 'INSERT INTO Annotations (slide, agreedClass, type) VALUES (%d,%d,%d)' % (slideUID,classID, typeId)
#        query = 'INSERT INTO Annotations (coordinateX1, coordinateY1, coordinateX2, coordinateY2, slide, class1, person1) VALUES (%d,%d,%d,%d,%d,%d, %d)' % (x1,y1,x2,y2,slideUID,classID,annotator)
        self.execute(query)
        query = 'SELECT last_insert_rowid()'
        self.execute(query)
        annoId = self.fetchone()[0]
        if (typeId==2):
            self.annotations[annoId] = rectangularAnnotation(annoId, x1,y1,x2,y2)
        else:
            self.annotations[annoId] = circleAnnotation(annoId, x1,y1,x2,y2)
        
        self.appendToMinMaxCoordsList(self.annotations[annoId])

        annoList = np.array([[x1,y1,1],[x2,y2,2]])
        self.insertCoordinates(np.array(annoList), slideUID, annoId)
        
        self.addAnnotationLabel(classId=classID, person=annotator, annoId=annoId)

        self.commit()
        return annoId

    def setLastModified(self, annoid:int, lastModified:float):
        self.execute(f'UPDATE Annotations SET lastModified="{lastModified}" where uid={annoid}')
        self.annotations[annoid].lastModified = lastModified

    def setGUID(self, annoid:int, guid:str):
        self.execute(f'UPDATE Annotations SET guid="{guid}" where uid={annoid}')
        self.guids[guid]=annoid
    
    def last_inserted_id() -> int:
            query = 'SELECT last_insert_rowid()'
            self.execute(query)
            return self.fetchone()[0]


    def insertNewSpotAnnotation(self,xpos_orig,ypos_orig, slideUID, classID, annotator, type = 1):

        if (type == 4):
            query = 'INSERT INTO Annotations (slide, agreedClass, type) VALUES (%d,0,%d)' % (slideUID, type)
            self.execute(query)
            query = 'SELECT last_insert_rowid()'
            self.execute(query)
            annoId = self.fetchone()[0]

            self.insertCoordinates(np.array([[xpos_orig, ypos_orig,1]]), slideUID, annoId)
            self.annotations[annoId] = spotAnnotation(annoId, xpos_orig,ypos_orig, (type==4))

        else:
            query = 'INSERT INTO Annotations (slide, agreedClass, type) VALUES (%d,%d,%d)' % (slideUID,classID, type)
            self.execute(query)
            query = 'SELECT last_insert_rowid()'
            self.execute(query)
            annoId = self.fetchone()[0]

            self.insertCoordinates(np.array([[xpos_orig, ypos_orig,1]]), slideUID, annoId)
            self.execute(query)

            self.annotations[annoId] = spotAnnotation(annoId, xpos_orig,ypos_orig, (type==4))
            self.addAnnotationLabel(classId=classID, person=annotator, annoId=annoId)

        self.appendToMinMaxCoordsList(self.annotations[annoId])
        self.commit()
        return annoId

    def removeFileFromDatabase(self, fileUID:int):
        self.execute('DELETE FROM Annotations_label where annoID IN (SELECT uid FROM Annotations where slide == %d)' % fileUID)
        self.execute('DELETE FROM Annotations_coordinates where annoID IN (SELECT uid FROM Annotations where slide == %d)' % fileUID)
        self.execute('DELETE FROM Annotations where slide == %d' % fileUID)
        self.execute('DELETE FROM Slides where uid == %d' % fileUID)
        self.commit()

    def removeAnnotationLabel(self, labelIdx, annoIdx):
            q = 'DELETE FROM Annotations_label WHERE uid == %d' % labelIdx
            self.execute(q)
            self.annotations[annoIdx].removeLabel(labelIdx)
            self.commit()
            self.checkCommonAnnotation(annoIdx)

    def changeAnnotationID(self, annoId:int, newAnnoID:int):
            self.execute('SELECT COUNT(*) FROM Annotations where uid == (%d)' % newAnnoID)
            if (self.fetchone()[0] > 0):
                return False

            self.execute('UPDATE Annotations_label SET annoId = %d WHERE annoId == %d' % (newAnnoID,annoId))
            self.execute('UPDATE Annotations_coordinates SET annoId = %d WHERE annoId == %d' % (newAnnoID,annoId))
            self.execute('UPDATE Annotations SET uid= %d WHERE uid == %d ' % (newAnnoID,annoId))
            self.annotations[annoId].uid = newAnnoID
            self.annotations[newAnnoID] = self.annotations[annoId]
            self.annotations.pop(annoId)
            self.appendToMinMaxCoordsList(self.annotations[newAnnoID])
            self.commit()
            return True


    def removeAnnotation(self, annoId, onlyMarkDeleted:bool=True):
            if (onlyMarkDeleted):
                self.execute('UPDATE Annotations SET deleted=1 WHERE uid == '+str(annoId))
                self.annotations[annoId].deleted = 1
            else:
                self.execute('DELETE FROM Annotations_label WHERE annoId == %d' % annoId)
                self.execute('DELETE FROM Annotations_coordinates WHERE annoId == %d' % annoId)
                self.execute('DELETE FROM Annotations WHERE uid == '+str(annoId))
                self.annotations.pop(annoId)
            self.commit()
            

    def insertNewSlide(self,slidename:str,slidepath:str,uuid:str=""):
            if (len(slidepath.split(os.sep))>1):
                directory = slidepath.split(os.sep)[-2]
            else:
                directory = ''
            self.execute('INSERT INTO Slides (filename,directory,uuid) VALUES ("%s","%s", "%s")' % (slidename,directory,uuid))
            self.commit()


    def fetchall(self):
        if (self.isOpen()):
            return self.dbcur.fetchall()
        else:
            print('Warning: DB not opened for fetch.')

    def fetchone(self):
        if (self.isOpen()):
            return self.dbcur.fetchone()
        else:
            print('Warning: DB not opened for fetch.')

    def execute(self, query):
        if (self.isOpen()):
            return self.dbcur.execute(query)
        else:
            print('Warning: DB not opened.')

    def getDBname(self):
        if (self.dbOpened):
            return self.dbname
        else:
            return ''
    
    def getAnnotatorByID(self, id):
        if (id is None):
            return ''
        self.execute('SELECT name FROM Persons WHERE uid == %d' % id)
        fo = self.fetchone()
        if (fo is not None):
            return fo[0]
        else:
            return ''

    def getClassByID(self, id):
        try:
            self.execute('SELECT name FROM Classes WHERE uid == %d' % id)
            return self.fetchone()[0]
        except:
            return '-unknown-'+str(id)+'-'


    def getAllPersons(self):
        self.execute('SELECT name, uid FROM Persons ORDER BY uid')
        return self.fetchall()

    def getAllClasses(self):
        self.execute('SELECT name,uid FROM Classes ORDER BY uid')
        return self.fetchall()
    
    def renameClass(self, classID, name):
        self.execute('UPDATE Classes set name="%s" WHERE uid ==  %d' % (name, classID))
        self.commit()

    def commit(self):
        return self.db.commit()

    def countEntryPerClass(self, slideID = 0):
        retval = {'unknown' :  {'uid': 0, 'count_total':0, 'count_slide':0}}
        self.dbcur.execute('SELECT Classes.uid, COUNT(*), name FROM Annotations LEFT JOIN Classes on Classes.uid == Annotations.agreedClass GROUP BY Classes.uid')
        allClasses = self.dbcur.fetchall()
    
        classids = np.zeros(len(allClasses))        
        for idx,element in enumerate(allClasses):
                name = element[2] if element[2] is not None else 'unknown'
                
                retval[name] = {'uid': element[0], 'count_total':element[1], 'count_slide':0}
        uidToName = {uid:name for uid,_,name in allClasses}


        if (slideID is not None):

            self.dbcur.execute('SELECT Classes.uid, COUNT(*), name FROM Annotations LEFT JOIN Classes on Classes.uid == Annotations.agreedClass where slide==%d GROUP BY Classes.uid' % slideID)
            allClasses = self.dbcur.fetchall()
            
            for uid,cnt,name in allClasses:
                name = 'unknown' if name is None else name
                retval[name]['count_slide'] = cnt

        return retval


    def countEntries(self):
        self.dbcur.execute('SELECT COUNT(*) FROM Annotations')
        num1 = self.dbcur.fetchone()

        return num1[0]

    def fetchSpotAnnotation(self,entryId=0):
        self.dbcur.execute('SELECT coordinateX, coordinateY FROM Annotations_coordinates WHERE annoId ==  '+str(entryId))
        coords = self.dbcur.fetchone()
        
        class1,class2,person1,person2=None,None,None,None
        self.dbcur.execute('SELECT Classes.name FROM Annotations_label LEFT JOIN Classes on Annotations_label.class == Classes.uid WHERE Annotations_label.annoId ==  '+str(entryId))
        classes = self.dbcur.fetchall()

        self.dbcur.execute('SELECT Persons.name FROM Annotations_label LEFT JOIN Persons on Annotations_label.person == Persons.uid WHERE Annotations_label.annoId ==  '+str(entryId))
        persons = self.dbcur.fetchall()

        return coords, classes, persons

    def fetchAreaAnnotation(self,entryId=0):
        self.dbcur.execute('SELECT coordinateX, coordinateY FROM Annotations_coordinates WHERE annoId ==  '+str(entryId)+' ORDER BY Annotations_coordinates.orderIdx')
        coords1 = self.dbcur.fetchone()
        coords2 = self.dbcur.fetchone()
        
        class1,class2,person1,person2=None,None,None,None
        self.dbcur.execute('SELECT Classes.name FROM Annotations_label LEFT JOIN Classes on Annotations_label.class == Classes.uid WHERE Annotations_label.annoId ==  '+str(entryId))
        classes = self.dbcur.fetchall()

        self.dbcur.execute('SELECT Persons.name FROM Annotations_label LEFT JOIN Persons on Annotations_label.person == Persons.uid WHERE Annotations_label.annoId ==  '+str(entryId))
        persons = self.dbcur.fetchall()

        return coords1, coords2, classes, persons

    

    def create(self,dbfilename) -> bool:
 
        if (os.path.isfile(dbfilename)):
             # ok, remove old file
            os.remove(dbfilename)

        try:
            tempdb = sqlite3.connect(dbfilename)
        except sqlite3.OperationalError:
            return False

        tempcur = tempdb.cursor()

        tempcur.execute('CREATE TABLE `Annotations_label` ('
            '	`uid`	INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,'
            '	`person`	INTEGER,'
            '	`class`	INTEGER,'
            '	`annoId`	INTEGER'
            ');')
        
        tempcur.execute('CREATE TABLE `Annotations_coordinates` ('
            '`uid`	INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,'
            '`coordinateX`	INTEGER,'
            '`coordinateY`	INTEGER,'
            '`slide`	INTEGER,'
            '`annoId`	INTEGER,'
            '`orderIdx`	INTEGER'
            ');')

        tempcur.execute('CREATE TABLE `Annotations` ('
            '`uid`	INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,'
            '`slide`	INTEGER,'
           	'`guid`	TEXT,'
            f'`lastModified`	REAL DEFAULT {time.time()},'
         	'`deleted`	INTEGER DEFAULT 0,'
            '`type`	INTEGER,'
            '`agreedClass`	INTEGER'
            ');')

        tempcur.execute('CREATE TABLE `Classes` ('
            '`uid`	INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,'
            '`name`	TEXT'
            ');')
        
        tempcur.execute('CREATE TABLE `Persons` ('
            '`uid`	INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,'
            '`name`	TEXT'
            ');')

        tempcur.execute('CREATE TABLE `Slides` ('
            '`uid`	INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,'
            '`filename`	TEXT,'
            '`width`	INTEGER,'
            '`height`	INTEGER,'
            '`directory` TEXT,'
            '`uuid` TEXT' 
            ');')

        tempdb.commit()
        self.db = tempdb
        self.dbcur = self.db.cursor()
        self.dbcur.execute('PRAGMA user_version = 1')
        self.db.create_function("generate_uuid",0, generate_uuid)
        self.db.create_function("pycurrent_time",0, time.time)
        self.dbfilename = dbfilename
        self.dbcur.execute(self.databaseStructure['Log'].getCreateStatement())
        self.annotations = dict()
        self.dbname = os.path.basename(dbfilename)
        self.dbOpened=True
        self.generateMinMaxCoordsList()
        self.addTriggers()

        return self
