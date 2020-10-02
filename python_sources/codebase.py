import os
import io
import sys
from contextlib import contextmanager
from time import time
from tqdm import tqdm
import numpy as np
import math
import json

import torch
from torch import cuda
from torch.utils.model_zoo import load_url
from skimage.segmentation import slic
import shap

import matplotlib.pyplot as plt
from IPython.display import display, Image as InteractiveImage
from PIL import Image
import imageio


VERSION = "1.4.8"
"""Notes:
- added separate colorbars per subfig
"""

print("Using 'codebase.py' version", VERSION)

# -----------------------------------------------------------------------------


class Timer:
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("{} time: {:.3f} s".format(self.title, time() - self.start_time))

# -----------------------------------------------------------------------------


class VideoLoader:

    DFDC_path = "../input/deepfake-detection-challenge/"
    DFDC_metadataPath = DFDC_path + "train_sample_videos/metadata.json"
    DFDC_trainVideoDir = DFDC_path + "train_sample_videos/"
    DFDC_testVideoDir = DFDC_path + "test_videos/"
    MDFD_testVideoDir = "../input/mesonet-dataset-sfw/validation/"

    def npSeqFromDir(
            directory: str, targetSize: tuple = None,
            normalize=True, frameLimit=np.infty) -> np.ndarray:

        # take only the files for which the extension represents an image
        imageList = [img for img in sorted(os.listdir(
            directory)) if img[-4:].lower() in [".jpg", ".jpeg", ".png"]]

        if (not targetSize):
            img = Image.open(os.path.join(directory, imageList[0]))
            targetSize = (img.width, img.height)

        sequence = np.zeros((min(len(imageList), frameLimit),
                             targetSize[0], targetSize[1], 3), dtype="uint8")

        for idx, filename in enumerate(imageList):
            # check the list length
            if (idx >= frameLimit):
                break
            # load the image with PIL
            img = Image.open(os.path.join(directory, filename))
            # resize if necessary
            imgSize = (img.width, img.height)
            if (imgSize != targetSize):
                img = img.resize(targetSize)
            # add the image to the list
            sequence[idx, :, :, :] = np.array(img)

        if (normalize):
            # return a numpy 4D array with values in [0,1]
            return sequence / 255.
        else:
            # return a numpy 4D array with values in [0,255]
            return sequence

    def loadFilenamesDFDC(
            videoCount=10, fakeClassValue=1, realClassValue=0):
        # Load metadata file containing labels for videos ("REAL" or "FAKE")
        with open(VideoLoader.DFDC_metadataPath) as f:
            metadata = json.load(f)

        videosInDirectory = [vid for vid in sorted(os.listdir(
            VideoLoader.DFDC_trainVideoDir)) if vid[-4:].lower() == ".mp4"]
        vidList = []
        for vidname in videosInDirectory[: min(videoCount,
                                               len(videosInDirectory))]:
            if (metadata[vidname]["label"] == "FAKE"):
                sequence_value = fakeClassValue
            else:
                sequence_value = realClassValue
            vidList.append((vidname, sequence_value))

        print("Total video names extracted from DFDC:", len(vidList))
        return vidList

    def loadDirnamesMDFD(
            label="df", videoCount=10, fakeClassValue=1, realClassValue=0):
        # label: "real" or "df"
        videosInDirectory = sorted(os.listdir(
            os.path.join(VideoLoader.MDFD_testVideoDir, label)))
        vidList = []
        for vidname in videosInDirectory[: min(videoCount,
                                               len(videosInDirectory))]:
            if (label == "df"):
                sequence_value = fakeClassValue
            else:
                sequence_value = realClassValue
            vidList.append((vidname, sequence_value))
        print("Total video names extracted from MDFD:", len(vidList))
        return vidList

# -----------------------------------------------------------------------------


class Segmenter:

    def __init__(self, mode="color", segmentsNumber=100, segCompactness=20):
        # mode : "color", "grid2D"
        self.mode = mode
        self.segmentsNumber = segmentsNumber
        self.segCompactness = segCompactness

    def segment(self, media: np.ndarray) -> np.ndarray:
        if (self.mode == "color"):
            return self._colorSegmentation(media)
        elif (self.mode == "grid2D"):
            return self._gridSegmentation(media)
        else:
            raise Exception(
                f"Segmentation mode not recognized: {self.mode}. "
                "Choose one between 'color' and 'grid2D'.")

    def _colorSegmentation(self, media):
        # if it's an image
        if (len(media.shape) == 3):
            return slic(
                media, n_segments=self.segmentsNumber,
                compactness=self.segCompactness, sigma=3)
        # if it's a sequence
        elif (len(media.shape) == 4):
            # sigma parameters is the size of the gaussian filter that
            # pre-smooths the data. I defined the sigma as a triplet where the
            # dimensions represent (time, image_x, image_y)
            return slic(
                media, n_segments=self.segmentsNumber,
                compactness=self.segCompactness, sigma=(0.5, 3, 3))
        else:
            raise Exception(f"Media shape not recognized: {media.shape}")

    def _gridSegmentation(self, media):
        if (not Segmenter._isSquare(self.segmentsNumber)):
            raise Exception("Segments number must be a perfect square.")

        if (len(media.shape) == 3):
            return self._gridSegmentationFrame(media)

        elif (len(media.shape) == 4):
            sliced = np.zeros((media.shape[:3]), dtype=np.int64)
            for idx, frame in enumerate(media):
                sliced[idx] = self._gridSegmentationFrame(frame)
            return sliced

        else:
            raise Exception(f"Media shape not recognized: {media.shape}")

    def _gridSegmentationFrame(self, frame):
        cellsPerEdge = int(math.sqrt(self.segmentsNumber))

        cellSize = (
            math.ceil(frame.shape[0] / cellsPerEdge),
            math.ceil(frame.shape[0] / cellsPerEdge))
        slicedFrame = np.zeros(
            (frame.shape[0], frame.shape[1]), dtype=np.int64)

        for i in range(cellsPerEdge):
            xCoords1 = i*cellSize[0]
            xCoords2 = min((i+1)*cellSize[0], frame.shape[0])
            for j in range(cellsPerEdge):
                yCoords1 = j*cellSize[1]
                yCoords2 = min((j+1)*cellSize[1], frame.shape[1])
                slicedFrame[xCoords1:xCoords2,
                            yCoords1:yCoords2] = i*cellsPerEdge + j
        return slicedFrame

    def _isSquare(apositiveint):
        # code from:
        # https://stackoverflow.com/questions/2489435/check-if-a-number-is-a-perfect-square
        x = apositiveint // 2
        seen = set([x])
        while x * x != apositiveint:
            x = (x + (apositiveint // x)) // 2
            if x in seen:
                return False
            seen.add(x)
        return True

# -----------------------------------------------------------------------------


# if GPU is available, import the numpy accelerated module
GPU = cuda.is_available()
if (GPU):
    try:
        import cupy as cp
    except ImportError:
        result = os.system('pip install cupy-cuda101')
        if (result == 0):
            import cupy as cp
        else:
            raise Exception("Could not pip install CuPy.")
    print("Using GPU acceleration for Numpy")


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class Explainer:

    def __init__(self, classifier, trackTime=False):
        # modelName : "mesonet" or "icpr"
        self.classifier = classifier
        self.model = classifier.getModel()
        self.classifierName = classifier.NAME
        self.trackTime = trackTime

        # visualization: make a color map
        from matplotlib.colors import LinearSegmentedColormap
        colors = []
        fc = (.96, .15, .34)  # fake color (RGB): red
        rc = (.09, .77, .36)  # real color (RGB): green
        for alpha in np.linspace(1, 0, 100):
            colors.append((fc[0], fc[1], fc[2], alpha))
        for alpha in np.linspace(0, 1, 100):
            colors.append((rc[0], rc[1], rc[2], alpha))
        self.cm = LinearSegmentedColormap.from_list("shap", colors)

    def normalizePredictions(self, p):
        # the predictions are normalized so that FAKE = -1 and REAL = +1
        a = self.classifier.FAKE_CLASS_VAL
        b = self.classifier.REAL_CLASS_VAL
        return 2 * (p - (a+b)/2) / (b-a)

    def explain(
            self, media: np.ndarray, segmentation: np.ndarray,
            nSegments, shapSamples: int) -> tuple:

        # in order to use the GPU and increase performance, numpy arrays are
        # converted at the beginning to cupy arrays
        if (GPU):
            cp_media = cp.asarray(media)
            cp_segmentation = cp.asarray(segmentation)
        else:
            cp_media = media
            cp_segmentation = segmentation

        # prediction function to pass to SHAP
        def f(z):
            # z: feature vectors from the point of view of shap
            #    from our point of view they are binary vectors defining which
            #    segments are active in the image.
            p = self.predictSamples(z, cp_media, cp_segmentation)
            return self.normalizePredictions(p)

        # use Kernel SHAP to explain the network's predictions
        # https://shap.readthedocs.io/en/latest/#shap.KernelExplainer
        background_data = np.zeros((1, nSegments))
        # A vector of features on which to explain the model’s output.
        samples_features = np.ones(nSegments)
        explainer = shap.KernelExplainer(f, background_data)
        shap_values = explainer.shap_values(
            samples_features, nsamples=shapSamples, l1_reg="aic")

        return shap_values[0], explainer.expected_value[0]

    def predictSamples(self, maskingPatterns, media, segments_slic):
        hideProgressBar = (maskingPatterns.shape[0] <= 1 or self.trackTime)

        predictions = []

        mask_time = 0
        pred_time = 0

        # if it's an image
        if (len(media.shape) == 3):
            avg = media.mean((0, 1))

            # create batches of masking patterns (for performance reasons)
            batchSize = 50
            batches = []
            i = 0
            while (i < maskingPatterns.shape[0]):
                if (i+batchSize < maskingPatterns.shape[0]):
                    j = i + batchSize
                else:
                    j = maskingPatterns.shape[0]
                batches.append(maskingPatterns[i:j, :])
                i += batchSize

            for batch in batches:

                # create masked images for this batch
                start_mask_time = time()
                masked_images_batch = []
                for maskingPattern in batch:
                    masked_images_batch.append(
                        Explainer.mask_image(maskingPattern, segments_slic,
                                             media, avg))
                mask_time += (time() - start_mask_time)

                # predict masked images for this batch
                start_pred_time = time()
                if (self.classifierName == "mesonet"):
                    preds = self.model.predict(
                        masked_images_batch)[0]
                elif (self.classifierName == "icpr"):
                    preds = self.classifier.predictFaceImages(
                        masked_images_batch)
                pred_time += (time() - start_pred_time)

                # concatenate this predictions with previous batch predictions
                predictions += list(preds)

            if (self.trackTime and maskingPatterns.shape[0] > 1):
                print("--- Masking:      %s seconds ---" % (mask_time))
                print("--- Predicting:   %s seconds ---" % (pred_time))

        # if it's a sequence
        elif (len(media.shape) == 4):
            avg = media.mean((0, 1, 2))
            for idx, maskingPattern in tqdm(enumerate(maskingPatterns),
                                            disable=hideProgressBar):

                start_mask_time = time()
                masked_sequence = Explainer.mask_sequence(
                    maskingPattern, segments_slic, media, avg)
                mask_time += (time() - start_mask_time)

                start_pred_time = time()
                if (self.classifierName == "mesonet"):
                    frames_preds = self.classifier.predict(
                        np.array(masked_sequence, ndmin=4))
                elif (self.classifierName == "icpr"):
                    frames_preds = self.classifier.predictFaceImages(
                        masked_sequence)
                video_pred = np.mean(frames_preds)
                pred_time += (time() - start_pred_time)

                predictions.append(video_pred)

            if (self.trackTime and maskingPatterns.shape[0] > 1):
                print("--- Masking:      %s seconds ---" % (mask_time))
                print("--- Predicting:   %s seconds ---" % (pred_time))

        return np.array(predictions, ndmin=2)
        # Predictions should be a numpy array like
        # [[0.6044281 ] [0.6797433 ] [0.5042769 ] ... ]
        # or
        # [[0.49638838 0.99638945 0.42781693 ... ]]

    def mask_image(
            mask_pattern, segmentation, image, background=None
            ) -> np.ndarray:
        # mask_pattern : a binary array having length 'nSegments'
        # segmentation : a 3D cupy or numpy array containing the segmentId
        #                of every pixel
        # image        : a 3D cupy or numpy array containing the image
        if background is None:
            background = image.mean((0, 1))

        if (GPU):
            out = cp.zeros(image.shape[0:3])
        else:
            out = np.zeros(image.shape[0:3])
        out[:, :, :] = image

        for j, segm_state in enumerate(mask_pattern):
            if (segm_state == 0):
                out[segmentation == j, :] = background
        if (GPU):
            return cp.asnumpy(out)
        else:
            return out

    def mask_sequence(
            mask_pattern, segmentation, sequence, background=None
            ) -> np.ndarray:
        # mask_pattern : a binary array having length 'nSegments'
        # segmentation : a 4D cupy or numpy array containing the segmentId
        #                of every pixel
        # image        : a 4D cupy or numpy array containing the image sequence
        if background is None:
            background = sequence.mean((0, 1, 2))

        if (GPU):
            out = cp.zeros(sequence.shape[0:4])
        else:
            out = np.zeros(sequence.shape[0:4])
        out[:, :, :, :] = sequence

        for j, segm_state in enumerate(mask_pattern):
            if (segm_state == 0):
                out[segmentation == j, :] = background
        if (GPU):
            return cp.asnumpy(out)
        else:
            return out

    def getExplanationFigure(
            self, img, image_true_class, prediction,
            shap_values, shap_values_time_avg, shap_values_std,
            segments_slic):
        # shap values : a numpy array of length N_SEGMENTS

        # Fill every segment with the color relative to the shapley value
        def fill_segmentation(values, segmentation):
            out = np.zeros(segmentation.shape)
            for i in range(len(values)):
                out[segmentation == i] = values[i]
            return out

        def subplot_with_title(ax, img, title="", alpha=1):
            if (img is not None):
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        def overlay_shap_values(ax, shap_values, segments_slic):
            m = fill_segmentation(shap_values, segments_slic)
            max_val = np.max(np.abs(shap_values))
            return ax.imshow(
                m, cmap=self.cm, vmin=-max_val, vmax=max_val)

        def overlay_shap_std(ax, shap_std, segments_slic):
            m = fill_segmentation(shap_std, segments_slic)
            max_val = np.max(np.abs(shap_std))
            return ax.imshow(m, vmin=0, vmax=max_val)

        def addColorbar(fig, ax, im, label):
            cb = fig.colorbar(im, ax=ax, label=label,
                              orientation="horizontal", aspect=10)
            cb.outline.set_visible(False)

        if (img.dtype != "uint8"):
            raise Exception(
                "getExplanationFigure(): 'img' numpy array "
                "must be of type 'uint8'")
        if (len(shap_values_time_avg) != len(shap_values)):
            raise Exception(
                "getExplanationFigure(): len(shap_values) and "
                "len(shap_values_time_avg) must be the same.")
        if (len(shap_values_std) != len(shap_values)):
            raise Exception(
                "getExplanationFigure(): len(shap_values) and "
                "len(shap_values_std) must be the same.")

        fig, axes = plt.subplots(
            nrows=1, ncols=4, figsize=(12, 4), constrained_layout=True)

        # 1st image (original image)

        subplot_with_title(
            axes[0], img, title="class: {}, pred: {:.3f}".format(
                self.normalizePredictions(image_true_class),
                self.normalizePredictions(prediction)))

        # 2nd image (gray images)

        gray_image = np.mean(img, axis=2).astype("uint8")
        gray_image = np.dstack((gray_image, gray_image, gray_image))
        subplot_with_title(axes[1], gray_image, alpha=0.15,
                           title="fake = red | real = green")
        # set up segments color overlay
        im = overlay_shap_values(axes[1], shap_values, segments_slic)
        addColorbar(fig, axes[1], im, "SHAP value")

        # 3rd image (gray images)

        subplot_with_title(axes[2], np.array([[[255, 255, 255]]]), alpha=0.15,
                           title="Time-averaged")
        # set up segments color overlay
        im = overlay_shap_values(axes[2], shap_values_time_avg, segments_slic)
        addColorbar(fig, axes[2], im, "SHAP value")

        # 4th image

        subplot_with_title(axes[3], np.array([[[255, 255, 255]]]), alpha=0.15,
                           title="Standard deviation")
        im = overlay_shap_std(axes[3], shap_values_std, segments_slic)
        addColorbar(fig, axes[3], im, "Value standard deviation")

        return fig

# -----------------------------------------------------------------------------


class FigureManager:

    def fig2pil():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im.show()
        buf.close()
        return im

    def fig2arrayRGB():
        return np.array(FigureManager.fig2pil())[:, :, :3]

    def saveAndDisplayGIF(
            sequence, outputName="sequence.gif",
            fps=5, displayOnNotebook=True):
        outputPath = f"../working/{outputName}"
        images = []
        for frame in sequence:
            images.append(Image.fromarray(frame))
        images[0].save(outputName, save_all=True, append_images=images[1:],
                       duration=1000/fps, loop=0)
        """with imageio.get_writer(outputName, mode='I', fps=fps) as writer:
            for frame in sequence:
                writer.append_data(frame)"""
        if (displayOnNotebook):
            display(InteractiveImage(outputPath))

    def saveAndDisplayImages(
            images, outputPrefix="output", displayOnNotebook=True):
        for i, image in enumerate(images):
            outputPath = f"../working/{outputPrefix}_{i}.jpg"
            with imageio.get_writer(outputPath, mode='i') as writer:
                writer.append_data(image)
            if (displayOnNotebook):
                display(Image.open(outputPath))

    def saveAverageSequence(sequence, outputName="avg_sequence.png"):
        avg_array = np.mean(np.array(sequence), axis=0)
        # Round values in array and cast as 8-bit integer
        avg_array = np.array(np.round(avg_array), dtype=np.uint8)
        avg_img = Image.fromarray(avg_array, mode="RGB")
        avg_img.save(outputName)

# -----------------------------------------------------------------------------


if (os.path.exists("../input/mesonet")):
    print("Mesonet module detected.")
    sys.path.append("../input/")
    from mesonet.classifiers import MesoInception4
else:
    print("Mesonet module not detected.")


class Mesonet:

    NAME = "mesonet"
    INPUT_SIZE = 256
    # output values for the classifier
    REAL_CLASS_VAL = 1
    FAKE_CLASS_VAL = 0

    def __init__(self):
        # Load classifier and weights
        self.classifier = MesoInception4()
        self.classifier.load("../input/mesonet/weights/MesoInception_DF")

    def getModel(self):
        return self.classifier.model

# -----------------------------------------------------------------------------


if (os.path.exists('../input/icpr2020')):
    print("ICPR2020 module detected.")
    result = os.system('pip install efficientnet-pytorch')
    if (result != 0):
        raise Exception("Could not pip install EfficientNet.")
    import sys
    sys.path.append('../input/icpr2020')
    from blazeface import FaceExtractor, BlazeFace, VideoReader
    from architectures import fornet, weights
    from isplutils import utils
else:
    print("ICPR2020 module not detected.")


class ICPR:

    NAME = "icpr"
    INPUT_SIZE = 224
    # output values for the classifier
    REAL_CLASS_VAL = 0
    FAKE_CLASS_VAL = 1

    def __init__(self, frames_per_video=10, consecutive_frames=False):
        # Choose an architecture between:
        # "EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4",
        # "EfficientNetAutoAttB4ST", "Xception"
        net_model = "EfficientNetAutoAttB4"

        # Choose a training dataset between:
        # DFDC, FFPP
        train_db = "DFDC"

        self.device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        face_policy = 'scale'

        model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
        self.net = getattr(fornet, net_model)().eval().to(self.device)
        self.net.load_state_dict(
            load_url(model_url, map_location=self.device, check_hash=True))

        self.transf = utils.get_transformer(
            face_policy, self.INPUT_SIZE,
            self.net.get_normalizer(), train=False)

        facedet = BlazeFace().to(self.device)
        facedet.load_weights("../input/icpr2020/blazeface/blazeface.pth")
        facedet.load_anchors("../input/icpr2020/blazeface/anchors.npy")
        videoreader = VideoReader(verbose=False)

        def video_read_fn_spaced(path):
            return videoreader.read_frames(path, num_frames=frames_per_video)

        def video_read_fn_conseq(path):
            # readapted from blazeface code (VideoReader)
            import cv2
            capture = cv2.VideoCapture(path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if (frame_count <= 0):
                return None
            frame_idxs = range(min(frames_per_video, frame_count))
            result = videoreader._read_frames_at_indices(
                path, capture, frame_idxs)
            capture.release()
            return result

        if (consecutive_frames):
            self.face_extractor = FaceExtractor(
                video_read_fn=video_read_fn_conseq, facedet=facedet)
        else:
            self.face_extractor = FaceExtractor(
                video_read_fn=video_read_fn_spaced, facedet=facedet)

    def getModel(self):
        return self.net

    def getFaceCroppedImages(self, images):
        faceList = []
        for image in images:
            faceImages = self.face_extractor.process_image(img=image)
            # take the face with the highest confidence found by BlazeFace
            if (faceImages['faces']):
                faceList.append(faceImages['faces'][0])
        return np.array(faceList)

    def getFaceCroppedVideo(self, videoPath):
        faceList = self.face_extractor.process_video(videoPath)
        faceList = [np.array(frame['faces'][0])
                    for frame in faceList if len(frame['faces'])]
        sequence = np.zeros((len(faceList), self.INPUT_SIZE,
                             self.INPUT_SIZE, 3), dtype="uint8")
        for idx, face in enumerate(faceList):
            # resize the image
            sequence[idx, :, :, :] = np.array(Image.fromarray(
                face).resize((self.INPUT_SIZE, self.INPUT_SIZE)))
        return sequence

    def predictFaceImages(self, images):
        faces_t = torch.stack([self.transf(image=img)['image']
                               for img in images])
        with torch.no_grad():
            raw_preds = self.net(faces_t.to(self.device))
            faces_pred = torch.sigmoid(raw_preds).cpu().numpy().flatten()
        return faces_pred

    def predictImages(self, images):
        faceList = self.getFaceCroppedImages(images)
        faces_pred = self.predictFaceImages(faceList)
        return faces_pred

# -----------------------------------------------------------------------------


class Pipeline:
    def __init__(self, classifier, seg: Segmenter, expl: Explainer,
                 segmentationDim: str, explanationMode: str,
                 nSegments: int, shapSamples: int, displayIFigures=True):
        # segmentationDim : a string among ["2D", "3D"]
        # explanationMode : a string among ["frame", "video"]
        self.classifier = classifier
        self.seg = seg
        self.expl = expl
        self.segmentationDim = segmentationDim
        self.explanationMode = explanationMode
        self.nSegments = nSegments
        self.shapSamples = shapSamples
        self.displayIFigures = displayIFigures
        self.shapValuesCollection = {}

    def start(self, imageSequence: np.ndarray, sequenceClass, sequenceName):
        if (not isinstance(imageSequence, np.ndarray)
                or len(imageSequence.shape) != 4):
            raise Exception(
                "imageSequence must be a 4-dimensional Numpy array.")

        # SEGMENT IMAGE OR VIDEO

        with Timer("Segmenting"):
            segmentation = self._segment(imageSequence)

        # COMPUTE SHAP VALUES

        with Timer("Computing SHAP values"):
            shap_values = self._explain(imageSequence, segmentation)

        shap_values_time_avg = np.mean(shap_values, axis=0)
        shap_values_std = np.std(shap_values, axis=0)

        # COMPUTE CLASSIFIER PREDICTIONS FOR FRAMES OR VIDEO

        with Timer("Classifier original predictions"):
            if (self.explanationMode == "video"):
                videoPred = self._predictSequence(imageSequence)
                # print("Normalized pred.: {:.3f}"
                #      " | Sum of shap values: {:.3f}".format(
                #        expl.normalizePredictions(videoPred),
                #        np.sum(shap_values)))
                framesPred = np.ones(imageSequence.shape[0]) * videoPred
                framesShapValues = np.tile(
                    shap_values, (imageSequence.shape[0], 1))
            elif (self.explanationMode == "frame"):
                framesPred = np.zeros(imageSequence.shape[0])
                for i in range(imageSequence.shape[0]):
                    framesPred[i] = self._predictSequence([imageSequence[i]])
                framesShapValues = shap_values

        # SHOW FIGURES

        figureInfo = [
            (sequenceName.split('.')[0]).replace('_', '-'),
            f"frames-{imageSequence.shape[0]}",
            f"segments-{self.segmentationDim}-{self.nSegments}",
            f"expl-{self.explanationMode}wise",
            f"samples-{self.shapSamples}"
        ]

        with Timer("Showing and saving figures"):
            fig_sequence = []
            for i in range(imageSequence.shape[0]):
                fig = self.expl.getExplanationFigure(
                    imageSequence[i], sequenceClass, framesPred[i],
                    framesShapValues[i], shap_values_time_avg, shap_values_std,
                    segmentation[i])
                fig.suptitle(", ".join(figureInfo) +
                             "\n"+"fake = -1 | real = +1")
                fig_sequence.append(FigureManager.fig2arrayRGB())
                plt.close('all')

            # Convert the list of frames into a 4D numpy array
            # (frame, width, height, color)
            fig_sequence = np.array(fig_sequence)

            # Example:
            # "obiwan_frames-20_segments-3D-100_expl-framewise_samples-500.gif"
            FigureManager.saveAndDisplayGIF(
                fig_sequence, outputName="_".join(figureInfo)+".gif",
                fps=1, displayOnNotebook=self.displayIFigures)
            if (imageSequence.shape[0] > 1):
                FigureManager.saveAverageSequence(
                    fig_sequence, outputName="_".join(figureInfo)+"_avg.png")

        self.shapValuesCollection["_".join(figureInfo)] = shap_values.tolist()

    def getShapValuesCollection(self):
        return self.shapValuesCollection

    def _segment(self, imageSequence):
        if (self.segmentationDim == "3D"):
            segmentation = self.seg.segment(imageSequence)
        elif (self.segmentationDim == "2D"):
            segmentation = np.zeros(imageSequence.shape[0:3])
            for i, frame in enumerate(imageSequence):
                segmentation[i, :, :] = self.seg.segment(frame)
        return segmentation

    def _explain(self, imageSequence, segmentation):
        if (self.explanationMode == "video"):
            with suppress_stderr():
                shap_values, expected_value = self.expl.explain(
                    imageSequence, segmentation,
                    self.nSegments, self.shapSamples)
            # print("Prediction on all-masked image |"
            #       " expected value: {:.3f}".format(expected_value))

        elif (self.explanationMode == "frame"):
            shap_values = np.zeros((imageSequence.shape[0], self.nSegments))
            for i in range(imageSequence.shape[0]):
                print(f"\rAnalizing frame: {i+1}/{imageSequence.shape[0]}",
                      end="" if (i+1) < imageSequence.shape[0] else "\n")
                with suppress_stderr():
                    shap_values[i], _ = self.expl.explain(
                        imageSequence[i], segmentation[i],
                        self.nSegments, self.shapSamples)

        return np.array(shap_values, ndmin=2)

    def _predictSequence(self, imageSequence):
        if (self.classifier.NAME == "icpr"):
            framePreds = self.classifier.predictFaceImages(imageSequence)
            videoPred = np.mean(framePreds)
        elif (self.classifier.NAME == "mesonet"):
            raise Exception("mesonet prediction not implemented yet.")
        return videoPred
