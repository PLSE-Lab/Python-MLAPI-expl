import pydicom
import numpy as np
from pathlib import Path
from random import randint
from PIL import ImageFile
from dicomslide import *
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.data_block import *
from tqdm import tqdm
from enum import Enum

def draw_rect(ax:plt.Axes, b:Collection[int], color:str='white', text=None, text_size=14):
    "Draw bounding box on `ax`."
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    _draw_outline(patch, 4)
    if text is not None:
        patch = ax.text(*b[:2], text, verticalalignment='top', color=color, fontsize=text_size, weight='bold')
        _draw_outline(patch,1)
        

def show_results(img, bbox_pred, preds, scores, classes, bbox_gt, preds_gt, figsize=(5,5)
                 , titleA: str="", titleB: str="", titleC: str=None, clas_pred=None, anchors=None, cla2_pred=None):

    if anchors is not None:
        sizes = len(np.unique(anchors[:,2]))

    
    cols = 2 if clas_pred is None else 2+sizes
    _, ax = plt.subplots(nrows=1, ncols=cols, figsize=figsize)
    ax[0].set_title(titleA)
    ax[1].set_title(titleB)
    if titleC is not None:
        ax[2].set_title(titleC)
        sizes = len(np.unique(anchors[:,2]))


    # show prediction
    img.show(ax=ax[1])
    if bbox_pred is not None:
        for bbox, c, scr in zip(bbox_pred, preds, scores):
            txt = str(c.item()) if classes is None else classes[c.item()]
            if (bbox.shape[-1]==4):
                draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr:.2f}') 

    # show gt
    img.show(ax=ax[0])
    for bbox, c in zip(bbox_gt, preds_gt):
        txt = str(c.item()) if classes is None else classes[c.item()]
        if (bbox.shape[-1] == 4):
            draw_rect(ax[0], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt}')


    if (clas_pred is not None):
        pred_act = clas_pred[:,0] # only CAM for active class, not for BG
        splits=1
        newshape=np.int16(np.sqrt(anchors.shape[0]/splits/sizes)),np.int16(np.sqrt(anchors.shape[0]/splits/sizes))

        for i in range(sizes):
                im = ax[i+2].imshow(np.reshape(torch.sigmoid(pred_act[i::sizes]).cpu(), newshape=newshape), vmin=0, vmax=1)
                plt.colorbar(im, ax=ax[i+2])




def create_anchors(sizes, ratios, scales, flatten=True):
    "Create anchor of `sizes`, `ratios` and `scales`."
    aspects = [[[s*math.sqrt(r), s*math.sqrt(1/r)] for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).view(-1,2)
    anchors = []
    for h,w in sizes:
        #4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h,2/w])).unsqueeze(0)
        base_grid = create_grid((h,w)).unsqueeze(1)
        n,a = base_grid.size(0),aspects.size(0)
        ancs = torch.cat([base_grid.expand(n,a,2), sized_aspects.expand(n,a,2)], 2)
        anchors.append(ancs.view(h,w,a,4))
    return torch.cat([anc.view(-1,4) for anc in anchors],0) if flatten else anchors


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EveryPointInterpolation = 1
    ElevenPointInterpolation = 2


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    Relative = 1
    Absolute = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    GroundTruth = 1
    Detected = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH
    or (X1,Y1,X2,Y2) => XYX2Y2

        Developed by: Rafael Padilla
        Last modification: May 24 2018
    """
    XYWH = 1
    XYX2Y2 = 2


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convertToRelativeValues(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convertToAbsoluteValues(size, box):
    # w_box = round(size[0] * box[2])
    # h_box = round(size[1] * box[3])
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)


def add_bb_into_image(image, bb, color=(255, 0, 0), thickness=2, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    x1, y1, x2, y2 = bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


class BoundingBox:
    def __init__(self,
                 imageName,
                 classId,
                 x,
                 y,
                 w,
                 h,
                 typeCoordinates=CoordinatesType.Absolute,
                 imgSize=None,
                 bbType=BBType.GroundTruth,
                 classConfidence=None,
                 format=BBFormat.XYWH):
        """Constructor.
        Args:
            imageName: String representing the image name.
            classId: String value representing class id.
            x: Float value representing the X upper-left coordinate of the bounding box.
            y: Float value representing the Y upper-left coordinate of the bounding box.
            w: Float value representing the width bounding box.
            h: Float value representing the height bounding box.
            typeCoordinates: (optional) Enum (Relative or Absolute) represents if the bounding box
            coordinates (x,y,w,h) are absolute or relative to size of the image. Default:'Absolute'.
            imgSize: (optional) 2D vector (width, height)=>(int, int) represents the size of the
            image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
            bbType: (optional) Enum (Groundtruth or Detection) identifies if the bounding box
            represents a ground truth or a detection. If it is a detection, the classConfidence has
            to be informed.
            classConfidence: (optional) Float value representing the confidence of the detected
            class. If detectionType is Detection, classConfidence needs to be informed.
            format: (optional) Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the
            coordinates of the bounding boxes. BBFormat.XYWH: <left> <top> <width> <height>
            BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        self._imageName = imageName
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if bbType == BBType.Detected and classConfidence is None:
            raise IOError(
                'For bbType=\'Detection\', it is necessary to inform the classConfidence value.')
        # if classConfidence != None and (classConfidence < 0 or classConfidence > 1):
        # raise IOError('classConfidence value must be a real value between 0 and 1. Value: %f' %
        # classConfidence)

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        self._format = format

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (typeCoordinates == CoordinatesType.Relative):
            (self._x, self._y, self._w, self._h) = convertToAbsoluteValues(imgSize, (x, y, w, h))
            self._width_img = imgSize[0]
            self._height_img = imgSize[1]
            if format == BBFormat.XYWH:
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = x
            self._y = y
            if format == BBFormat.XYWH:
                self._w = w
                self._h = h
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = w
                self._y2 = h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        if imgSize is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = imgSize[0]
            self._height_img = imgSize[1]

    def getAbsoluteBoundingBox(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def getRelativeBoundingBox(self, imgSize=None):
        print('getting relative BBOX')
        if imgSize is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if imgSize is None:
            return convertToRelativeValues((imgSize[0], imgSize[1]),
                                           (self._x, self._y, self._w, self._h))
        else:
            return convertToRelativeValues((self._width_img, self._height_img),
                                           (self._x, self._y, self._w, self._h))

    def getImageName(self):
        return self._imageName

    def getConfidence(self):
        return self._classConfidence

    def getFormat(self):
        return self._format

    def getClassId(self):
        return self._classId

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates

    def getBBType(self):
        return self._bbType

    @staticmethod
    def compare(det1, det2):
        det1BB = det1.getAbsoluteBoundingBox()
        det1ImgSize = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2ImgSize = det2.getImageSize()

        if det1.getClassId() == det2.getClassId() and \
           det1.classConfidence == det2.classConfidenc() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1ImgSize[0] == det1ImgSize[0] and \
           det2ImgSize[1] == det2ImgSize[1]:
            return True
        return False

    @staticmethod
    def clone(boundingBox):
        absBB = boundingBox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        newBoundingBox = BoundingBox(
            boundingBox.getImageName(),
            boundingBox.getClassId(),
            absBB[0],
            absBB[1],
            absBB[2],
            absBB[3],
            typeCoordinates=boundingBox.getCoordinatesType(),
            imgSize=boundingBox.getImageSize(),
            bbType=boundingBox.getBBType(),
            classConfidence=boundingBox.getConfidence(),
            format=BBFormat.XYWH)
        return newBoundingBox
    
class BoundingBoxes:
    def __init__(self):
        self._boundingBoxes = []

    def addBoundingBox(self, bb):
        self._boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if BoundingBox.compare(d, _boundingBox):
                del self._boundingBoxes[d]
                return

    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []
    
    def removeAllBoundingObjects(self):
        self.removeAllBoundingBoxes()

    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType]

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName]

    def count(self, bbType=None):
        if bbType is None:  # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType:  # get only specified bb type
                count += 1
        return count

    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    def drawAllBoundingBoxes(self, image, imageName):
        bbxes = self.getBoundingBoxesByImageName(imageName)
        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
        return image
    

def sampling_func(y, **kwargs):
    y_label = np.array(y[1])
    h, w = kwargs['size']

    _arbitrary_prob = 0.1
    _mit_prob = 0.5
    
    sample_prob = np.array([_arbitrary_prob, 1-_arbitrary_prob-_mit_prob, _mit_prob])
    
    case = np.random.choice(3, p=sample_prob)
    
    
    
    bg_label = [0] if y_label.dtype == np.int64 else ["bg"]
    classes = bg_label + kwargs['classes']
    level_dimensions = kwargs['level_dimensions']
    level = kwargs['level']
    if ('bg_label_prob' in kwargs):
        _bg_label_prob = kwargs['bg_label_prob']
        if (_bg_label_prob>1.0):
            raise ValueError('Probability needs to be <= 1.0.')
    else:
        _bg_label_prob = 0.0  # add a backgound label to sample complete random
    
    if ('strategy' in kwargs):
        _strategy = kwargs['strategy']
    else:
        _strategy = 'normal'
        
    if ('set' in kwargs):
        _set = kwargs['set']
    else:
        _set = 'training'

    if ('negative_class' in kwargs):
        _negative_class = kwargs['negative_class']
    else:
        _negative_class = 7 # hard examples

        
    _random_offset_scale = 0.5  # up to 50% offset to left and right of frame
    xoffset = randint(-w, w) * _random_offset_scale
    yoffset = randint(-h, h) * _random_offset_scale
    coords = np.array(y[0])

    slide_width, slide_height = level_dimensions[level]
    
    if (case==0):
        if (_set == 'training'): # sample on upper part of image
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h)
        elif (_set == 'validation'): # sample on lower part of image
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h) + int(slide_height/2)
        elif (_set == 'test'):
            xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), slide_height - h)
    if (case==2): # mitosis
        
        ids = y_label == 1

        if (_set == 'training'):
            ids[coords[:,1]>slide_height/2] = 0 # lower part not allowed
        elif (_set == 'validation'):
            ids[coords[:,1]<slide_height/2] = 0 # upper part not allowed

        if (np.count_nonzero(ids)<1):
            if (_set == 'training'): # sample on upper part of image
                xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h)
            elif (_set == 'validation'): # sample on lower part of image
                xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h) + int(slide_height/2)
        else:
            xmin, ymin, xmax, ymax = np.array(y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
    if (case==1): #nonmitosis
            annos = kwargs['annotations']
            coords = np.array(annos[_negative_class]['bboxes'])
            
            ids = np.arange(len(coords))

            if (_set == 'training'):
                ids[coords[:,1]>slide_height/2] = 0 # lower part not allowed
            elif (_set == 'validation'):
                ids[coords[:,1]<slide_height/2] = 0 # upper part not allowed

            if (np.count_nonzero(ids)<1):

                if (_set == 'training'): # sample on upper part of image
                    xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h)
                elif (_set == 'validation'): # sample on lower part of image
                    xmin, ymin = randint(int(w / 2 - xoffset), slide_width - w), randint(int(h / 2 - yoffset), int(slide_height/2) - h) + int(slide_height/2)
            else:
                xmin, ymin, xmax, ymax = coords[ids][randint(0, np.count_nonzero(ids) - 1)]
        
    return int(xmin - w / 2 + xoffset), int(ymin - h / 2 +yoffset)


class PascalVOCMetric(Callback):

    def __init__(self, anchors, size, metric_names: list, detect_thresh: float=0.3, nms_thresh: float=0.3
                 , images_per_batch: int=-1):
        self.ap = 'AP'
        self.anchors = anchors
        self.size = size
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh

        self.images_per_batch = images_per_batch
        self.metric_names_original = metric_names
        self.metric_names = ["{}-{}".format(self.ap, i) for i in metric_names]

        self.evaluator = Evaluator()
        self.boundingObjects = BoundingBoxes()


    def on_epoch_begin(self, **kwargs):
        self.boundingObjects.removeAllBoundingObjects()
        self.imageCounter = 0


    def on_batch_end(self, last_output, last_target, **kwargs):
#        print('Last target:',last_target)

        bbox_gt_batch, class_gt_batch = last_target[:2]
        class_pred_batch, bbox_pred_batch = last_output[:2]

        self.images_per_batch = self.images_per_batch if self.images_per_batch > 0 else class_pred_batch.shape[0]
        for bbox_gt, class_gt, clas_pred, bbox_pred in \
                list(zip(bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch))[: self.images_per_batch]:

            out = process_output(clas_pred, bbox_pred, self.anchors, self.detect_thresh)
            bbox_pred, scores, preds = out['bbox_pred'], out['scores'], out['preds']
            if bbox_pred is None:# or len(preds) > 3 * len(bbox_gt):
                continue

            #image = np.zeros((512, 512, 3), np.uint8)

            # if the number is to hight evaluation is very slow
            total_nms_examples = len(class_gt) * 3
            bbox_pred = bbox_pred[:total_nms_examples]
            scores = scores[:total_nms_examples]
            preds = preds[:total_nms_examples]
            to_keep = nms(bbox_pred, scores, self.nms_thresh)
            bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()

            t_sz = torch.Tensor([(self.size, self.size)])[None].cpu()
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0]
            # change gt from x,y,x2,y2 -> x,y,w,h
            if (bbox_gt.shape[-1] == 4):
                bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))
            bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
            # change from center to top left
            if (bbox_gt.shape[-1] == 4):
                bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

            class_gt = to_np(class_gt) - 1
            preds = to_np(preds)
            scores = to_np(scores)

            for box, cla in zip(bbox_gt, class_gt):
                    temp = BoundingBox(imageName=str(self.imageCounter), classId='Mit', x=box[0], y=box[1],
                                   w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                                   bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(self.size,self.size))

                    self.boundingObjects.addBoundingBox(temp)



            # to reduce math complexity take maximal three times the number of gt boxes
            num_boxes = len(bbox_gt) * 3
            for box, cla, scor in list(zip(bbox_pred, preds, scores))[:num_boxes]:
                    temp = BoundingBox(imageName=str(self.imageCounter), classId='Mit', x=box[0], y=box[1],
                                       w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute, classConfidence=scor,
                                       bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(self.size, self.size))

                    self.boundingObjects.addBoundingBox(temp)


            #image = self.boundingObjects.drawAllBoundingBoxes(image, str(self.imageCounter))
            self.imageCounter += 1

    def on_epoch_end(self, last_metrics, **kwargs):
        if self.boundingObjects.count() > 0:

            self.metrics = {}
            metricsPerClass = self.evaluator.GetPascalVOCMetrics(self.boundingObjects, IOUThreshold=0.3)
            self.metric = max(sum([mc[self.ap] for mc in metricsPerClass]) / len(metricsPerClass), 0)

            for mc in metricsPerClass:
                self.metrics['{}-{}'.format(self.ap, mc['class'])] = max(mc[self.ap], 0)

            return {'last_metrics': last_metrics + [self.metric]}
        else:
            self.metrics = dict(zip(self.metric_names, [0 for i in range(len(self.metric_names))]))
            return {'last_metrics': last_metrics + [0]}
        
        
        

def get_slides(slidelist_test:list, database:"Database", positive_class:int=2, negative_class:int=7, basepath:str='WSI', size:int=256):


    lbl_bbox=list()
    files=list()
    train_slides=list()
    val_slides=list()

    getslides = """SELECT uid, filename FROM Slides"""
    for idx, (currslide, filename) in enumerate(tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
        if (str(currslide) in slidelist_test): # skip test slides
            continue

        database.loadIntoMemory(currslide)

        slide_path = basepath + os.sep + filename

        slide = DicomSlide(str(slide_path))

        level = 0#slide.level_count - 1
        level_dimension = slide.level_dimensions[level]
        down_factor = slide.level_downsamples[level]

        classes = {positive_class: 1} # Map non-mitosis to background

        labels, bboxes = [], []
        annotations = dict()
        for id, annotation in database.annotations.items():
            annotation.r = 25
            d = 2 * annotation.r / down_factor
            x_min = (annotation.x1 - annotation.r) / down_factor
            y_min = (annotation.y1 - annotation.r) / down_factor
            x_max = x_min + d
            y_max = y_min + d
            if annotation.agreedClass not in annotations:
                annotations[annotation.agreedClass] = dict()
                annotations[annotation.agreedClass]['bboxes'] = list()
                annotations[annotation.agreedClass]['label'] = list()

            annotations[annotation.agreedClass]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
            annotations[annotation.agreedClass]['label'].append(annotation.agreedClass)

            if annotation.agreedClass in classes:
                label = classes[annotation.agreedClass]

                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                labels.append(label)

        if len(bboxes) > 0:
            lbl_bbox.append([bboxes, labels])
            files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='training', negative_class=negative_class)))
            train_slides.append(len(files)-1)

            lbl_bbox.append([bboxes, labels])
            files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size, y=[bboxes, labels], sample_func=partial(sampling_func, set='validation', negative_class=negative_class)))
            val_slides.append(len(files)-1)

    return lbl_bbox, train_slides,val_slides,files


class SlideContainer():

    def __init__(self, file: Path, annotations:dict, y, level: int=0, width: int=256, height: int=256, sample_func: callable=None):
        self.file = file
        self.slide = DicomSlide(str(file))
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.y = y
        self.annotations = annotations
        self.sample_func = sample_func
        self.classes = list(set(self.y[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int=0, y: int=0):
             return np.array(self.slide.read_region(location=(int(x * self.down_factor),int(y * self.down_factor)),
                                          level=self.level, size=(self.width, self.height)))[:, :, :3]


    @property
    def shape(self):
        return (self.width, self.height)

    def __str__(self):
        return 'SlideContainer with:\n sample func: '+str(self.sample_func)+'\n slide:'+str(self.file)

    def get_new_train_coordinates(self):
        # use passed sampling method
        if callable(self.sample_func):
            return self.sample_func(self.y, **{"classes": self.classes, "size": self.shape,
                                               "level_dimensions": self.slide.level_dimensions,
                                               "annotations" : self.annotations,
                                               "level": self.level, "container" : self})

        # use default sampling method
        class_id = np.random.choice(self.classes, 1)[0]
        ids = self.y[1] == class_id
        xmin, ymin, _, _ = np.array(self.y[0])[ids][randint(0, np.count_nonzero(ids) - 1)]
        return int(xmin - self.shape / 2), int(ymin - self.height / 2)

def bb_pad_collate_min(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    samples = [s for s in samples if s[1].data[0].shape[0] > 0] # check that labels are available

    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls = s[1].data
        bboxes[i,-len(lbls):] = bbs
        labels[i,-len(lbls):] = torch.from_numpy(lbls)
    return torch.cat(imgs,0), (bboxes,labels)

class SlideLabelList(LabelList):


    def __getitem__(self,idxs:Union[int,np.ndarray])->'LabelList':
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            if self.item is None:
                slide_container = self.x.items[idxs]

                xmin, ymin = slide_container.get_new_train_coordinates()

                x = self.x.get(idxs, xmin, ymin)
                y = self.y.get(idxs, xmin, ymin)
            else:
                x,y = self.item ,0
            if self.tfms or self.tfmargs:
                x = x.apply_tfms(self.tfms, **self.tfmargs)
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve':False})
            if y is None: y=0
            return x,y
        else:
            return self.new(self.x[idxs], self.y[idxs])

        

def nms(boxes, scores, thresh=0.5):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    to_keep, indexes = [], torch.LongTensor(range_of(scores))
    while len(scores) > 0:
        #pdb.set_trace()
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals <= thresh
        if len(mask_keep.nonzero()) == 0: break
        idx_first = mask_keep.nonzero().min().item()
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    return LongTensor(to_keep)

def rescale_boxes(bboxes, t_sz: Tensor):
    t_sz = t_sz.to(bboxes.device)
    if (bboxes.shape[-1] == 4):
        bboxes[:, 2:] = bboxes[:, 2:] * t_sz / 2
        bboxes[:, :2] = (bboxes[:, :2] + 1) * t_sz / 2
    else:
        bboxes[:, 2:] = bboxes[:, 2:] * t_sz[...,0] / 2
        bboxes[:, :2] = (bboxes[:, :2] + 1) * t_sz / 2

    return bboxes

def activ_to_bbox(acts, anchors, flatten=True):
    "Extrapolate bounding boxes on anchors from the model activations."

    if flatten:

        if (anchors.shape[-1]==4):
            acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
            centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
            sizes = anchors[...,2:] * torch.exp(acts[...,2:])

        else:
            acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2]]))
            centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
            sizes = anchors[...,2:] * torch.exp(acts[...,2:])
        return torch.cat([centers, sizes], -1)
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res
def process_output(clas_pred, bbox_pred, anchors, detect_thresh=0.25, use_sigmoid=True):
    bbox_pred = activ_to_bbox(bbox_pred, anchors.to(clas_pred.device))

    if (use_sigmoid):
        clas_pred = torch.sigmoid(clas_pred)

    clas_pred_orig = clas_pred.clone()
    detect_mask = clas_pred.max(1)[0] > detect_thresh
    if np.array(detect_mask.cpu()).max() == 0:
        return {'bbox_pred':None, 'scores':None, 'preds':None, 'clas_pred':clas_pred,'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}

    bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
    if (bbox_pred.shape[-1] == 4):
        bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
    else:
        bbox_pred = bbox_pred# torch.clamp(bbox_pred, min=-1, max=1)

        
    scores, preds = clas_pred.max(1)
    return {'bbox_pred':bbox_pred, 'scores':scores, 'preds':preds, 'clas_pred':clas_pred, 'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}

        
class Evaluator:
    def GetPascalVOCMetrics(self,
                            boundingboxes,
                            IOUThreshold=0.5,
                            method=MethodAveragePrecision.EveryPointInterpolation):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Get
        Args:
            boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
            (default value = 0.5);
            method (default = EveryPointInterpolation): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
            interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EveryPointInterpolation"  (ElevenPointInterpolation);
        Returns:
            A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        ret = []  # list containing metrics (precision, recall, average precision) of each class
        # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
        groundTruths = []
        # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        if ('getBoundingBoxes' in dir(boundingboxes)):
            for bb in boundingboxes.getBoundingBoxes():
                # [imageName, class, confidence, (bb coordinates XYX2Y2)]
                if bb.getBBType() == BBType.GroundTruth:
                    groundTruths.append([
                        bb.getImageName(),
                        bb.getClassId(), 1,
                        bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                    ])
                else:
                    detections.append([
                        bb.getImageName(),
                        bb.getClassId(),
                        bb.getConfidence(),
                        bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                    ])
                # get class
                if bb.getClassId() not in classes:
                    classes.append(bb.getClassId())
        else:
            for bb in boundingboxes.getBoundingObjects():
                # [imageName, class, confidence, (bb coordinates XYX2Y2)]
                if bb.getBBType() == BBType.GroundTruth:
                    groundTruths.append([
                        bb.getImageName(),
                        bb.getClassId(), 1,
                        bb.getAbsoluteBoundingObject(BBFormat.XYX2Y2)
                    ])
                else:
                    detections.append([
                        bb.getImageName(),
                        bb.getClassId(),
                        bb.getConfidence(),
                        bb.getAbsoluteBoundingObject(BBFormat.XYX2Y2)
                    ])
                # get class
                if bb.getClassId() not in classes:
                    classes.append(bb.getClassId())

        
        classes = sorted(classes)
        # Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in groundTruths if g[1] == c]
            npos = len(gts)
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            FN = np.zeros(len(gts))
            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key, val in det.items():
                det[key] = np.zeros(val)
            # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
            # Loop through detections
            for d in tqdm(range(len(dects)),desc='Detections'):
                # print('dect %s => %s' % (dects[d][0], dects[d][3],))
                # Find ground truth image
                gt = [gt for gt in gts if gt[0] == dects[d][0]]
                iouMax = sys.float_info.min
                for j in range(len(gt)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iouMax >= IOUThreshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d] = 1  # count as true positive
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                        # print("TP")
                    else:
                        FP[d] = 1  # count as false positive
                        # print("FP")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            
            for gt in tqdm(range(len(gts)), desc='GT'):

                d = [d for d in dects if d[0] == gts[gt][0]]

                iouMax = sys.float_info.min
                # find maximum IOU
                for j in range(len(d)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = Evaluator.iou(gts[gt][3], d[j][3])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iouMax < IOUThreshold:
                    FN[gt] = 1  # count as false negative
                    # print("FP")


                
            # compute precision, recall and average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EveryPointInterpolation:
                [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'F1': (2*np.sum(TP)/(2*np.sum(TP)+np.sum(FN)+np.sum(FP))),
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
                'total FN': np.sum(FN),
            }
            ret.append(r)
        return ret



    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    # 11-point interpolated average precision
    def ElevenPointInterpolatedAP(rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]

    # For each detections, calculate IOU with reference
    @staticmethod
    def _getAllIOUs(reference, detections):
        ret = []
        bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        # img = np.zeros((200,200,3), np.uint8)
        for d in detections:
            bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            iou = Evaluator.iou(bbReference, bb)
            # Show blank image with the bounding boxes
            # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
            # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
            ret.append((iou, reference, d))  # iou, reference, detection
        # cv2.imshow("comparing",img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("comparing")
        return sorted(ret, key=lambda i: i[0], reverse=True)  # sort by iou (from highest to lowest)

    @staticmethod
    def iou(objA, objB):
        if (len(objA)==4):
            # if boxes dont intersect
            if Evaluator._boxesIntersect(objA, objB) is False:
                return 0
            interArea = Evaluator._getIntersectionArea(objA, objB)
            union = Evaluator._getUnionAreas(objA, objB, interArea=interArea)
            # intersection over union
            iou = interArea / union
        else: # circular objects
            if Evaluator._circlesIntersect(objA, objB) is False:
                return 0
            distance = np.sqrt(np.square(objA[0]-objB[0])+np.square(objA[1]-objB[1]))

            radius1 = objA[2]
            radius2 = objB[2]
            acosterm1 = np.arccos((((distance**2) + (radius1**2) - (radius2**2)) / (2 * distance * radius1 + 1e-8)).clip(-1,1))
            acosterm2 = np.arccos((((distance**2) - (radius1**2) + (radius2**2)) / (2 * distance * radius2 + 1e-8)).clip(-1,1))
            secondterm = np.sqrt(((radius1+radius2-distance)*(distance+radius1-radius2)*(distance+radius1+radius2)*(distance-radius1+radius2)).clip(min=0))

            intersec = (radius1**2 * acosterm1) + (radius2**2 * acosterm2) - (0.5 * secondterm)

            union = np.pi * ((radius1**2) + (radius2**2)) - intersec
            iou = intersec / (union+1e-8)
            
        #assert iou >= 0
        return max(iou,0)

    @staticmethod
    def _circlesIntersect(circA, circB):
         distance = np.sqrt(np.square(circA[0]-circB[0])+np.square(circA[1]-circB[1]))
         if distance>(circA[2]+circB[2]):
              return False
         else:
              return True

    
    
    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


PreProcessors = Union[PreProcessor, Collection[PreProcessor]]
fastai_types[PreProcessors] = 'PreProcessors'

def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:,:2] - boxes[:,2:]/2
    bot_right = boxes[:,:2] + boxes[:,2:]/2
    return torch.cat([top_left, bot_right], 1)

def bbox_to_activ(bboxes, anchors, flatten=True):

    "Return the target of the model on `anchors` for the `bboxes`."
    if flatten:
        # x and y offsets are normalized by radius
        t_centers = (bboxes[...,:2] - anchors[...,:2]) / anchors[...,2:]
        # Radius is given in log scale, relative to anchor radius
        t_sizes = torch.log(bboxes[...,2:] / anchors[...,2:] + 1e-8)
        # Finally, everything is divided by 0.1 (radii by 0.2)
        if (bboxes.shape[-1] == 4):
            return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
        else:
            return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2]]))

            
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res



def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a,t,4), tgts.unsqueeze(0).expand(a,t,4)
    top_left_i = torch.max(ancs[...,:2], tgts[...,:2])
    bot_right_i = torch.min(ancs[...,2:], tgts[...,2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[...,0] * sizes[...,1]
   
    
def IoU_values(anchors, targets):
    "Compute the IoU values of `anchors` by `targets`."
    inter = intersection(anchors, targets)
    anc_sz, tgt_sz = anchors[:,2] * anchors[:,3], targets[:,2] * targets[:,3]
    union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter

    return inter/(union+1e-8)

def create_grid(size):
    "Create a grid of a given `size`."
    H, W = size if is_tuple(size) else (size,size)
    grid = FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W, W) if W > 1 else tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H, H) if H > 1 else tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1,2)


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    "Match `anchors` to targets. -1 is match to background, -2 is ignore."
    ious = IoU_values(anchors, targets)
    matches = anchors.new(anchors.size(0)).zero_().long() - 2

    if ious.shape[1] > 0:
        vals,idxs = torch.max(ious,1)
        matches[vals < bkg_thr] = -1
        matches[vals > match_thr] = idxs[vals > match_thr]
    #Overwrite matches with each target getting the anchor that has the max IoU.
    #vals,idxs = torch.max(ious,0)
    #If idxs contains repetition, this doesn't bug and only the last is considered.
    #matches[idxs] = targets.new_tensor(list(range(targets.size(0)))).long()
    return matches

def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:,:2] + boxes[:,2:])/2
    sizes = boxes[:,2:] - boxes[:,:2]
    return torch.cat([center, sizes], 1)

def show_anchors_on_images(data, anchors, figsize=(15,15)):
    all_boxes = []
    all_labels = []
    x, y = data.one_batch(DatasetType.Train, True, True)
    for image, bboxes, labels in zip(x, y[0], y[1]):
        image = Image(image.float().clamp(min=0, max=1))

        # 0=not found; 1=found; found 2=anchor
        processed_boxes = []
        processed_labels = []
            
        for gt_box in tlbr2cthw(bboxes[labels > 0]) if (bboxes.shape[-1]==4) else bboxes[labels > 0]:
            matches = match_anchors(anchors, gt_box[None, :])
            bbox_mask = matches >= 0
            if bbox_mask.sum() != 0:
                bbox_tgt = anchors[bbox_mask]

                processed_boxes.append(to_np(gt_box))
                processed_labels.append(2)
                for bb in bbox_tgt:
                    processed_boxes.append(to_np(bb))
                    processed_labels.append(3)
            else:
                processed_boxes.append(to_np(gt_box))
                processed_labels.append(0)
                val, idx = torch.max(IoU_values(anchors, gt_box[None, :]), 0)
                best_fitting_anchor = anchors[idx][0]
                processed_boxes.append(to_np(best_fitting_anchor))
                processed_labels.append(1)

        all_boxes.extend(processed_boxes) 
        all_labels.extend(processed_labels)

        processed_boxes = np.array(processed_boxes)
        processed_labels = np.array(processed_labels)

        _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax[0].set_title("Matched Anchors")
        ax[1].set_title("No match")

        if sum(processed_labels == 2) > 0:
            if (processed_boxes.shape[-1] == 4):
                imageBB = ImageBBox.create(*image.size, cthw2tlbr(tensor(processed_boxes[processed_labels > 1])),
                                               labels=processed_labels[processed_labels > 1],
                                               classes=["", "", "Match", "Anchor"], scale=False)

                
            image.show(ax=ax[0], y=imageBB)
        else:
            image.show(ax=ax[0])

        if sum(processed_labels == 0) > 0:
            if (processed_boxes.shape[-1] == 4):
                imageBBNoMatch = ImageBBox.create(*image.size, cthw2tlbr(tensor(processed_boxes[processed_labels <= 1])),
                                                      labels=processed_labels[processed_labels <= 1],
                                                      classes=["No Match", "Anchor"], scale=False)

            image.show(ax=ax[1], y=imageBBNoMatch)
        else:
            image.show(ax=ax[1])


    return np.array(all_boxes), np.array(all_labels)



class SlideItemList(ItemList):

    def __init__(self, items:Iterator, path:PathOrStr='.', label_cls:Callable=None, inner_df:Any=None,
                 processor:PreProcessors=None, x:'ItemList'=None, ignore_empty:bool=False):
        self.path = Path(path)
        self.num_parts = len(self.path.parts)
        self.items,self.x,self.ignore_empty = items,x,ignore_empty
        self.sizes = [None] * len(self.items)
        if not isinstance(self.items,np.ndarray): self.items = array(self.items, dtype=object)
        self.label_cls,self.inner_df,self.processor = ifnone(label_cls,self._label_cls),inner_df,processor
        self._label_list,self._split = SlideLabelList,ItemLists
        self.copy_new = ['x', 'label_cls', 'path']

    def __getitem__(self,idxs: int, x: int=0, y: int=0)->Any:
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            return self.get(idxs, x, y)
        else:
            return self.get(*idxs)

    def label_from_list(self, labels:Iterator, label_cls:Callable=None, **kwargs)->'LabelList':
        "Label `self.items` with `labels`."
        labels = array(labels, dtype=object)
        label_cls = self.get_label_cls(labels, label_cls=label_cls, **kwargs)
        y = label_cls(labels, path=self.path, **kwargs)
        res = SlideLabelList(x=self, y=y)
        return res


class SlideImageItemList(SlideItemList):
    pass

class SlideObjectItemList(SlideImageItemList, ImageList):

    def get(self, i, x: int, y: int):
        fn = self.items[i]
        res = self.open(fn, x, y)
        self.sizes[i] = res.size
        return res

class ObjectItemListSlide(SlideObjectItemList):

    def open(self, fn: SlideContainer,  x: int=0, y: int=0):
        return Image(pil2tensor(fn.get_patch(x, y) / 255., np.float32))


class SlideObjectCategoryList(ObjectCategoryList):

    def get(self, i, x: int=0, y: int=0):
        h, w = self.x.items[i].shape
        bboxes, labels = self.items[i]
        if x > 0 and y > 0:
            bboxes = np.array(bboxes)
            labels = np.array(labels)

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

            bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
            bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

            ids = ((bboxes[:, 0] + bb_widths) > 0) \
                  & ((bboxes[:, 1] + bb_heights) > 0) \
                  & ((bboxes[:, 2] - bb_widths) < w) \
                  & ((bboxes[:, 3] - bb_heights) < h)

            bboxes = bboxes[ids]
            bboxes = np.clip(bboxes, 0, x)
            bboxes = bboxes[:, [1, 0, 3, 2]]

            labels = labels[ids]
            if len(labels) == 0:
                labels = np.array([0])
                bboxes = np.array([[0, 0, 1, 1]])

            return ImageBBox.create(h, w, bboxes, labels, classes=self.classes, pad_idx=self.pad_idx)
        else:
            return ImageBBox.create(h, w, bboxes[:10], labels[:10], classes=self.classes, pad_idx=self.pad_idx)


def slide_object_result(learn: Learner, anchors, detect_thresh:float=0.2, nms_thresh: float=0.3,  image_count: int=5):
    with torch.no_grad():
        img_batch, target_batch = learn.data.one_batch(DatasetType.Train, False, False, False)
        prediction_batch = learn.model(img_batch)
        class_pred_batch, bbox_pred_batch = prediction_batch[:2]
        regression_pred_batch = prediction_batch[3].view(-1) if len(prediction_batch) > 3 \
            else [None] * class_pred_batch.shape[0]
        bbox_regression_pred_batch = prediction_batch[4] if len(prediction_batch) > 4 \
            else [None] * bbox_pred_batch.shape[0]

        bbox_gt_batch, class_gt_batch = target_batch

        for img, bbox_gt, class_gt, clas_pred, bbox_pred, reg_pred, box_reg_pred in \
                list(zip(img_batch, bbox_gt_batch, class_gt_batch, class_pred_batch, bbox_pred_batch,
                         regression_pred_batch, bbox_regression_pred_batch))[:image_count]:
            img = Image(learn.data.denorm(img))

            out = process_output(clas_pred, bbox_pred, anchors, detect_thresh)
            bbox_pred, scores, preds = [out[k] for k in ['bbox_pred', 'scores', 'preds']]
            if bbox_pred is not None:
                to_keep = nms(bbox_pred, scores, nms_thresh)
                bbox_pred, preds, scores = bbox_pred[to_keep].cpu(), preds[to_keep].cpu(), scores[to_keep].cpu()
                box_reg_pred = box_reg_pred[to_keep].cpu() if box_reg_pred is not None else None

            t_sz = torch.Tensor([*img.size])[None].cpu()
            bbox_gt = bbox_gt[np.nonzero(class_gt)].squeeze(dim=1).cpu()
            class_gt = class_gt[class_gt > 0] - 1
            # change gt from x,y,x2,y2 -> x,y,w,h
            bbox_gt[:, 2:] = bbox_gt[:, 2:] - bbox_gt[:, :2]

            bbox_gt = to_np(rescale_boxes(bbox_gt, t_sz))
            if bbox_pred is not None:
                bbox_pred = to_np(rescale_boxes(bbox_pred, t_sz))
                # change from center to top left
                bbox_pred[:, :2] = bbox_pred[:, :2] - bbox_pred[:, 2:] / 2

            pred_score_classes = f'{np.mean(to_np(preds)):.2f}' if preds is not None else '0.0'
            pred_score_classes_reg = f'{np.mean(to_np(box_reg_pred)):.2f}' if box_reg_pred is not None else '0.0'
            gt_score = f'{np.mean(to_np(class_gt)):.2f}' if class_gt.shape[0] > 0 else '0.0'

            pred_score = '' if reg_pred is None else f'Box:{pred_score_classes} \n Reg:{to_np(reg_pred):.2f}'

            if box_reg_pred is None:
                show_results(img, bbox_pred, preds, scores, list(range(0, learn.data.c))
                             , bbox_gt, class_gt, (15, 3), titleA=str(gt_score), titleB=str(pred_score), titleC='CAM', clas_pred=clas_pred, anchors=anchors)
            else:
                pred_score_reg = f'BoxReg:{pred_score_classes_reg} \n Reg:{to_np(reg_pred):.2f}'

                show_results_with_breg(img, bbox_pred, preds, box_reg_pred, scores, list(range(0, learn.data.c))
                                       , bbox_gt, class_gt, (15, 15), titleA=str(gt_score), titleB=str(pred_score),
                                       titleC=pred_score_reg)

def _draw_outline(o:Patch, lw:int):
    "Outline bounding box onto image `Patch`."
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

                

def show_results_with_breg(img, bbox_pred, preds, scores, breg_pred, classes, bbox_gt, preds_gt, figsize=(5,5)
                 , titleA: str="", titleB: str="", titleC: str=""):

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    ax[0].set_title(titleA)
    ax[1].set_title(titleB)
    ax[2].set_title(titleC)

    # show gt
    img.show(ax=ax[0])
    for bbox, c in zip(bbox_gt, preds_gt):
        txt = str(c.item()) if classes is None else classes[c.item()]
        draw_rect(ax[0], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt}')

    # show prediction class
    img.show(ax=ax[1])
    if bbox_pred is not None:
        for bbox, c, scr in zip(bbox_pred, preds, scores):
            txt = str(c.item()) if classes is None else classes[c.item()]
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr.item():.1f}')

    # show prediction class
    img.show(ax=ax[2])
    if bbox_pred is not None:
        for bbox, c in zip(bbox_pred, breg_pred):
            draw_rect(ax[1], [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{c.item():.1f}')



