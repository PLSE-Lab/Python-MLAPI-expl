"""
Script with classes and functions to extract faces from videos for Deep Fake Detection Challenge
"""

import numpy as np
import os
import sys
import cv2
import PIL
import torch
import warnings

# ######################################################################################################
# Common Functions
def debug(*args):
    if VERBOSE == True:
        print(*args)

def basename_and_ext(filename):
    return os.path.splitext(filename)

# ######################################################################################################
# Face Detector
# 
# Facenet Pytorch Face Detector
try:
    from facenet_pytorch import MTCNN,extract_face
except ImportError as e:
    os.system('pip install facenet_pytorch')
    from facenet_pytorch import MTCNN,extract_face

def get_adjusted_box(box, h, w, margin=0.1):
    "Adjusts the face bounding box - adds margin and corrects out-of-bound coordinates"
    x1, y1, x2, y2 = box
    hf = y2 - y1
    wf = x2 - x1
    hf = int(hf * (1+margin))
    wf = int(wf * (1+margin))
    xmid = int((x2+x1) / 2)
    ymid = int((y2+y1) / 2)
    x2a = xmid  + int(wf / 2)
    x1a = xmid  - int(wf / 2)
    y2a = ymid  + int(hf / 2)
    y1a = ymid  - int(hf / 2)
    if (x1a < 0):
        x1a = 0
        x2a = wf
    if (y1a < 0):
        y1a = 0
        y2a = hf
    if (x2a > w):
        x2a = w
        x1a = w - wf
    if (y2a > h):
        y2a = h
        y1a = h - hf
    return x1a, y1a, x2a, y2a


class DFDCFaceDetector():
    "Face detector that uses facenet pytorch backend. Extend and replace for other backends."
    
    def __init__(self, min_face_size=20 ):
        """
        input: 
            min_face_size:  minimum height / width of detected face
        """
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.detector = MTCNN(device=device, min_face_size=min_face_size, keep_all=True, post_process=False)
    
    def detect_faces(self, frames, threshold=0.95):
        """
        Detects faces and returns a nested list of bounding boxes
        input: 
            frames: RGB Image Batch - batch_size * height * width * channels
            threshold: minimum probability at which a detection becomes a face
        output:
            nested list of face bounding boxes found in each frame, each box adjusted by margin
        """
        all_face_boxes = []
        for lb in np.arange(0, len(frames), 8):
            pil_frames = [PIL.Image.fromarray(frame) for frame in frames[lb:lb+8]]
            faces_by_frame,face_probs_by_frame = self.detector.detect(pil_frames)
            for face_boxes_in_frame, probs, pil_frame in zip(faces_by_frame,face_probs_by_frame, pil_frames):
                selected_faces_in_frame = []
                w, h = pil_frame.size
                if type(face_boxes_in_frame) != type(None):
                    for face_box, prob in zip(face_boxes_in_frame,probs):
                        if prob > threshold:
                            box = np.array(face_box).squeeze().astype(int)
                            selected_faces_in_frame.append(box)
                all_face_boxes.append(selected_faces_in_frame)
        return all_face_boxes

FACENET_DETECTOR = DFDCFaceDetector()

def _extract_faces_from_frame_list( frame_list, basename, detector:DFDCFaceDetector=FACENET_DETECTOR, faces_path=None, margin=0.1):
    "Extracts faces from video frames and optiionally saves them on disk"
    faces = []
    framenums = []
    try:
        face_boxes = detector.detect_faces(frame_list)
    except:
        print("Error detecting faces in video ", basename )
        return [], []
    framenum = 0
    for face_boxes_in_frame, frame in zip(face_boxes, frame_list):
        if type(face_boxes_in_frame) == type(None): #face detector returns none if frame has no face
            continue
        framenums.extend([framenum]*len(face_boxes_in_frame))
        h,w, _ = frame.shape
        facenum = 0
        for box in face_boxes_in_frame:
            x1a, y1a, x2a, y2a = get_adjusted_box(box, h, w, margin=margin)
            face = frame[y1a:y2a,x1a:x2a]
            face = PIL.Image.fromarray(face.astype('uint8'))
            if (faces_path != None):
                file_path = "%s/%s_%d_%d.jpg" % (faces_path, basename, framenum, facenum)
                face.save(file_path, quality=100, subsampling=0)
            facenum += 1
            faces.append(face)
        framenum+=1
    return faces, framenums

def _extract_face_pairs(real_frames, fake_frames, basename, detector:DFDCFaceDetector=FACENET_DETECTOR, margin=0.1):
    "Extracts real faces and corresponding fake faces from video frames."
    realfaces = []
    fakefaces = []
    framenums = []
    try:
        face_boxes = detector.detect_faces(real_frames)
    except:
        print("Error detecting faces in video ", basename )
        return [], [], []
    bbx_count = len(face_boxes)
    ff_count = len(fake_frames)
    rf_count = len(real_frames)
    min_count = min([bbx_count, ff_count, rf_count])

    framenum = 0
    for face_boxes_in_frame, fakeframe, realframe in zip(face_boxes[:min_count], fake_frames[:min_count], real_frames[:min_count]):
        if type(face_boxes_in_frame) == type(None): #face detector returns none if frame has no face
            continue
        framenums.extend([framenum]*len(face_boxes_in_frame))
        h,w, _ = realframe.shape
        hf,wf, _ = fakeframe.shape
        if hf != h or wf != w:
            print("Fake/Real frame dimensions mismatch - skipping...")
            continue
        for box in face_boxes_in_frame:
            x1a, y1a, x2a, y2a = get_adjusted_box(box, h, w, margin=margin)
            realface = realframe[y1a:y2a,x1a:x2a]
            realface = PIL.Image.fromarray(realface.astype('uint8'))
            realfaces.append(realface)
            fakeface = fakeframe[y1a:y2a,x1a:x2a]
            fakeface = PIL.Image.fromarray(fakeface.astype('uint8'))
            fakefaces.append(fakeface)
        framenum+=1
    return realfaces, fakefaces, framenums


# ######################################################################################################
# Image Extraction from Video using DALI (requires GPU)
# - DALI could not be used for inference as competition rules do not allow internet access
# - There appear to be memory leaks with DALI VideoPipeline it crashes with segmentation fault for some of the batches. My CV2 extracts ran on CPU took longer but always finished without errors.
# 
# https://github.com/NVIDIA/DALI/blob/master/docs/examples/sequence_processing/video/video_reader_simple_example.ipynb
try:
    from nvidia.dali.pipeline import Pipeline
except ImportError as e:
    os.system('pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali')
    from nvidia.dali.pipeline import Pipeline

import nvidia.dali.ops as ops
import nvidia.dali.types as types

class VideoPipe(Pipeline):
    "Sets up a video pipeline to read frames from a batch of videos using the NVIDIA DALI library"
    def __init__(self, num_threads, device_id, filenames, sequence_length, stride):
        super(VideoPipe, self).__init__(1, num_threads, device_id, seed=16)
        self.input = ops.VideoReader(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            stride=stride,
            random_shuffle=False)

    def define_graph(self):
        return self.input(name="Reader")

def _extract_frames_with_dali(video_paths, seq_length=10, stride=5):
    """
    Given a list of video file paths, extracts 'seq_length' image frames for each video at given stride
    Returns a nested list of RGB image frames, one for each video in 'video_paths' 
    """
    debug('-- processing DALI ' + str(video_paths[0]))
    gc.collect()
    skip = False
    vpipe = VideoPipe(num_threads=2, device_id=0, filenames=video_paths, sequence_length=seq_length, stride=stride)
    try:
        vpipe.build()
        frames = []
        for path in video_paths:
            pipe_out = vpipe.run()
            if (len(pipe_out) > 0):
                sequences_out = pipe_out[0].as_cpu().as_array()
                video_frames = sequences_out[0]
                video_frames = [f for f in video_frames if f.any()]
                frames.append(video_frames)
        debug('-- DALI', 'processed' + str(video_paths[0]), len(frames))
        del vpipe
        torch.cuda.empty_cache()
        gc.collect()
        return frames
    except:
        print("-- DALI Unexpected error for " + str(video_paths[0]) + ": ", sys.exc_info()[0])
        del vpipe
        torch.cuda.empty_cache()
        gc.collect()
        return []
    
# ######################################################################################################
# # Image Extraction from Video using CV2
def _extract_frames_with_cv2(video_path, seq_length=20, stride=2):
    "Extracts faces given path to the video file"
    captured = cv2.VideoCapture(video_path) 
    count = 0
    frame_list = []
    success  = captured.grab()
    k = 0
    while success == True and k < seq_length*stride :
        _, frame = captured.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame.copy())
        success = captured.grab()
        k+=stride
    captured.release()
    return frame_list

# ######################################################################################################
# Face Extraction from Video
class DFDCVideoFaceExtractor():
    "Extracts faces from video frames."
    
    def __init__(self, backend='CV2', detector:DFDCFaceDetector = FACENET_DETECTOR ):
        """
        input: 
            backend:  'CV2' is default, the other option is 'DALI'
        """
        self.backend = backend
        if backend == 'DALI':
            if not torch.cuda.is_available():
                warnings.warn('DALI requires GPU. Falling back to CV2...')
                self.backend = 'CV2'
        self.detector = detector
    
    def extract_frames(self, video_path, seq_length=5,stride=1):
        """
        Extracts frames from video file 
        input: 
            video_path: full path to the video file
            seq_length: number of frames to extract
            stride: gap between frames
        output:
            list of frames as numpy arrays (W x H x C) 
        """
        try:
            if self.backend == 'DALI':
                frames_list = _extract_frames_with_dali([video_path], seq_length=seq_length, stride=stride)
                if(len(frames_list) == 0 or len(frames_list[0]) == 0):
                    return [], [];

                frame_list = frames_list[0]
            else:
                frame_list = _extract_frames_with_cv2(video_path, seq_length=seq_length, stride=stride)

            return frame_list
        except:
            print(f"Error getting frames from {video_path}")
            return []
        
        
    def extract_faces(self, video_path,  seq_length=20, stride=2, faces_path=None, margin=0.1):
        """
        Extracts faces given path to the video file
        input: 
            video_path: full path to the video file
            seq_length: number of frames to extract
            stride: gap between frames
            output_path: (optional) path where faces are to be saved
            margin: proportion extra to grab around the detected face
        output:
            list of faces as numpy arrays (W x H x C) and corresponding frame numbers where they were found
            optionally saves face images to disk
        """
        try:
            frame_list = self.extract_frames(video_path, seq_length=seq_length, stride=stride)
            basename, _ = basename_and_ext(video_path.split('/')[-1])
            faces, framenums =  _extract_faces_from_frame_list(frame_list,
                                                              basename,
                                                              faces_path=faces_path,
                                                              detector = self.detector,
                                                              margin=margin)
            if (len(faces) == 0):
                print("no faces for " + basename)
                return [], []
            return faces, framenums
        except:
            print(f"Error reading faces from {video_path}")
            return [], []
        
    def extract_face_pairs(self, real_frames, fake_frames, basename, margin=0.1):
        """
        Detects faces in real frames and uses the real bounding boxes to extract image arrays
        from equivalent real and fake frames
        input: 
            real_frames: list of real frames
            fake_frames: list of real frames
            basename: basename for real video
            margin: proportion extra to grab around the detected face
        output:
            list of real and fake faces as numpy arrays (W x H x C) and corresponding frame numbers where they were found
        """
        return _extract_face_pairs(real_frames, fake_frames, basename, detector=self.detector, margin=margin)

    
    
