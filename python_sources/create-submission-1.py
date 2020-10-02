!apt-get -y install libportaudio2
!apt-get -y install libasound-dev
!apt-get -y install libportaudio2
!apt-get -y install libasound-dev
!pip install tensorflow-addons
!pip install keras-rectified-adam
!pip install SoundFile
!pip install SoundDevice

import numpy as np
import pickle
import librosa
import sounddevice as sd
import tensorflow as tf
import multiprocessing
import os
from sklearn.preprocessing import StandardScaler


import numpy as np
import pickle
import librosa
import tensorflow as tf
import pandas as pd
import os
import sounddevice as sd
import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import IPython.display as ipd
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
import multiprocessing

def inverse_stft_transform(stft_features, window_length, overlap):
    return librosa.istft(stft_features, win_length=window_length, hop_length=overlap)


def revert_features_to_audio(features, phase, window_length, overlap, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)
    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done

    features = np.transpose(features, (1, 0))
    return inverse_stft_transform(features, window_length=window_length, overlap=overlap)


def play(audio, sample_rate):
    # ipd.display(ipd.Audio(data=audio, rate=sample_rate))  # load a local WAV file
    sd.play(audio, sample_rate, blocking=True)


def add_noise_to_clean_audio(clean_audio, noise_signal):
    if len(clean_audio) >= len(noise_signal):
        # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noiseSegment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noiseSegment ** 2)
    noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
    return noisyAudio

def read_audio(filepath, sample_rate, normalize=True):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    if normalize is True:
        div_fac = 1 / np.max(np.abs(audio)) / 3.0
        audio = audio * div_fac
        # audio = librosa.util.normalize(audio)
    return audio, sr


def prepare_input_features(stft_features, numSegments, numFeatures):
    noisySTFT = np.concatenate([stft_features[:, 0:numSegments - 1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments, noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:, :, index] = noisySTFT[:, index:index + numSegments]
    return stftSegments


def get_input_features(predictorsList):
    predictors = []
    for noisy_stft_mag_features in predictorsList:
        # For CNN, the input feature consisted of 8 consecutive noisy
        # STFT magnitude vectors of size: 129 × 8,
        # TODO: duration: 100ms
        inputFeatures = prepare_input_features(noisy_stft_mag_features)
        # print("inputFeatures.shape", inputFeatures.shape)
        predictors.append(inputFeatures)

    return predictors


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_tf_feature(noise_stft_mag_features, clean_stft_magnitude, noise_stft_phase):
    noise_stft_mag_features = noise_stft_mag_features.astype(np.float32).tostring()
    clean_stft_magnitude = clean_stft_magnitude.astype(np.float32).tostring()
    noise_stft_phase = noise_stft_phase.astype(np.float32).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'noise_stft_phase': _bytes_feature(noise_stft_phase),
        'noise_stft_mag_features': _bytes_feature(noise_stft_mag_features),
        'clean_stft_magnitude': _bytes_feature(clean_stft_magnitude)}))
    return example

	
#utils over

#feature extractor
class FeatureExtractor:
    def __init__(self, audio, *, windowLength, overlap, sample_rate):
        self.audio = audio
        self.ffT_length = windowLength
        self.window_length = windowLength
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.window = scipy.signal.hamming(self.window_length, sym=False)

    def get_stft_spectrogram(self):
        return librosa.stft(self.audio, n_fft=self.ffT_length, win_length=self.window_length, hop_length=self.overlap,
                            window=self.window, center=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.window_length, hop_length=self.overlap,
                             window=self.window, center=True)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.audio, sr=self.sample_rate, power=2.0, pad_mode='reflect',
                                              n_fft=self.ffT_length, hop_length=self.overlap, center=True)

    def get_audio_from_mel_spectrogram(self, M):
        return librosa.feature.inverse.mel_to_audio(M, sr=self.sample_rate, n_fft=self.ffT_length,
                                                    hop_length=self.overlap,
                                                    win_length=self.window_length, window=self.window,
                                                    center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None)
													
#Dataset
class Dataset:
    def __init__(self, clean_filenames, noise_filenames, **config):
        self.clean_filenames = clean_filenames
        self.noise_filenames = noise_filenames
        self.sample_rate = config['fs']
        self.overlap = config['overlap']
        self.window_length = config['windowLength']
        self.audio_max_duration = config['audio_max_duration']

    def _sample_noise_filename(self):
        return np.random.choice(self.noise_filenames)

    def _remove_silent_frames(self, audio):
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=self.overlap, top_db=20)

        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

    def _phase_aware_scaling(self, clean_spectral_magnitude, clean_phase, noise_phase):
        assert clean_phase.shape == noise_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)

    def get_noisy_audio(self, *, filename):
        return read_audio(filename, self.sample_rate)

    def _audio_random_crop(self, audio, duration):
        audio_duration_secs = librosa.core.get_duration(audio, self.sample_rate)

        ## duration: length of the cropped audio in seconds
        if duration >= audio_duration_secs:
            # print("Passed duration greater than audio duration of: ", audio_duration_secs)
            return audio

        audio_duration_ms = math.floor(audio_duration_secs * self.sample_rate)
        duration_ms = math.floor(duration * self.sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return audio[idx: idx + duration_ms]

    def _add_noise_to_clean_audio(self, clean_audio, noise_signal):
        if len(clean_audio) >= len(noise_signal):
            # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
            while len(clean_audio) >= len(noise_signal):
                noise_signal = np.append(noise_signal, noise_signal)

        ## Extract a noise segment from a random location in the noise file
        ind = np.random.randint(0, noise_signal.size - clean_audio.size)

        noiseSegment = noise_signal[ind: ind + clean_audio.size]

        speech_power = np.sum(clean_audio ** 2)
        noise_power = np.sum(noiseSegment ** 2)
        noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
        return noisyAudio

    def parallel_audio_processing(self, clean_filename):

        clean_audio, _ = read_audio(clean_filename, self.sample_rate)

        # remove silent frame from clean audio
        clean_audio = self._remove_silent_frames(clean_audio)

        noise_filename = self._sample_noise_filename()

        # read the noise filename
        noise_audio, sr = read_audio(noise_filename, self.sample_rate)

        # remove silent frame from noise audio
        noise_audio = self._remove_silent_frames(noise_audio)

        # sample random fixed-sized snippets of audio
        clean_audio = self._audio_random_crop(clean_audio, duration=self.audio_max_duration)

        # add noise to input image
        noiseInput = self._add_noise_to_clean_audio(clean_audio, noise_audio)

        # extract stft features from noisy audio
        noisy_input_fe = FeatureExtractor(noiseInput, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        noise_spectrogram = noisy_input_fe.get_stft_spectrogram()

        # Or get the phase angle (in radians)
        # noisy_stft_magnitude, noisy_stft_phase = librosa.magphase(noisy_stft_features)
        noise_phase = np.angle(noise_spectrogram)

        # get the magnitude of the spectral
        noise_magnitude = np.abs(noise_spectrogram)

        # extract stft features from clean audio
        clean_audio_fe = FeatureExtractor(clean_audio, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        clean_spectrogram = clean_audio_fe.get_stft_spectrogram()
        # clean_spectrogram = cleanAudioFE.get_mel_spectrogram()

        # get the clean phase
        clean_phase = np.angle(clean_spectrogram)

        # get the clean spectral magnitude
        clean_magnitude = np.abs(clean_spectrogram)
        # clean_magnitude = 2 * clean_magnitude / np.sum(scipy.signal.hamming(self.window_length, sym=False))

        clean_magnitude = self._phase_aware_scaling(clean_magnitude, clean_phase, noise_phase)

        scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        noise_magnitude = scaler.fit_transform(noise_magnitude)
        clean_magnitude = scaler.transform(clean_magnitude)

        return noise_magnitude, clean_magnitude, noise_phase

    def create_tf_record(self, *, prefix, subset_size, parallel=True):
        counter = 0
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        for i in range(0, len(self.clean_filenames), subset_size):

            tfrecord_filename = prefix + '_' + str(counter) + '.tfrecords'

            if os.path.isfile(tfrecord_filename):
                print(f"Skipping {tfrecord_filename}")
                counter += 1
                continue

            writer = tf.io.TFRecordWriter(tfrecord_filename)
            clean_filenames_sublist = self.clean_filenames[i:i + subset_size]
            print(*clean_filenames_sublist)
            print(f"Processing files from: {i} to {i + subset_size}")
            if parallel:
                out = p.map(self.parallel_audio_processing, clean_filenames_sublist)
            else:
                out = [self.parallel_audio_processing(filename) for filename in clean_filenames_sublist]

            for o in out:
                noise_stft_magnitude = o[0]
                clean_stft_magnitude = o[1]
                noise_stft_phase = o[2]

                noise_stft_mag_features = prepare_input_features(noise_stft_magnitude, numSegments=8, numFeatures=129)

                noise_stft_mag_features = np.transpose(noise_stft_mag_features, (2, 0, 1))
                clean_stft_magnitude = np.transpose(clean_stft_magnitude, (1, 0))
                noise_stft_phase = np.transpose(noise_stft_phase, (1, 0))

                noise_stft_mag_features = np.expand_dims(noise_stft_mag_features, axis=3)
                clean_stft_magnitude = np.expand_dims(clean_stft_magnitude, axis=2)

                for x_, y_, p_ in zip(noise_stft_mag_features, clean_stft_magnitude, noise_stft_phase):
                    y_ = np.expand_dims(y_, 2)
                    example = get_tf_feature(x_, y_, p_)
                    writer.write(example.SerializeToString())

            counter += 1
            writer.close()
#Mozilla
class MozillaCommonVoiceDataset:

    def __init__(self, basepath, *, val_dataset_size):
        self.basepath = basepath
        self.val_dataset_size = val_dataset_size

    def _get_common_voice_filenames(self, dataframe_name='train.tsv'):
        mozilla_metadata = pd.read_csv(os.path.join(self.basepath, dataframe_name), sep='\t')
        clean_files = mozilla_metadata['path'].values
        np.random.shuffle(clean_files)
        print("Total number of training examples:", len(clean_files))
        return clean_files

    def get_train_val_filenames(self):
        clean_files = self._get_common_voice_filenames(dataframe_name='train.tsv')

        # resolve full path
        clean_files = [os.path.join(self.basepath, 'train',  filename) for filename in clean_files]

        clean_files = clean_files[:-self.val_dataset_size]
        clean_val_files = clean_files[-self.val_dataset_size:]
        print("# of Training clean files:", len(clean_files))
        print("# of  Validation clean files:", len(clean_val_files))
        return clean_files, clean_val_files


    def get_test_filenames(self):
        clean_files = self._get_common_voice_filenames(dataframe_name='test.tsv')

        # resolve full path
        clean_files = [os.path.join(self.basepath, 'test',  filename) for filename in clean_files]

        print("# of Testing clean files:", len(clean_files))
        return clean_files
#Urban
np.random.seed(999)


class UrbanSound8K:
    def __init__(self, basepath, *, val_dataset_size, class_ids=None):
        self.basepath = basepath
        self.val_dataset_size = val_dataset_size
        self.class_ids = class_ids

    def _get_urban_sound_8K_filenames(self):
        urbansound_metadata = pd.read_csv(os.path.join(self.basepath, 'UrbanSound8K.csv'))

        # shuffle the dataframe
        urbansound_metadata.reindex(np.random.permutation(urbansound_metadata.index))

        return urbansound_metadata

    def _get_filenames_by_class_id(self, metadata):

        if self.class_ids is None:
            self.class_ids = np.unique(metadata['classID'].values)
            print("Number of classes:", self.class_ids)

        all_files = []
        file_counter = 0
        for c in self.class_ids:
            per_class_files = metadata[metadata['classID'] == c][['slice_file_name', 'fold']].values
            per_class_files = [os.path.join(self.basepath,  'fold' + str(file[1]), file[0]) for file in
                               per_class_files]
            print("Class c:", str(c), 'has:', len(per_class_files), 'files')
            file_counter += len(per_class_files)
            all_files.extend(per_class_files)

        assert len(all_files) == file_counter
        return all_files

    def get_train_val_filenames(self):
        urbansound_metadata = self._get_urban_sound_8K_filenames()

        # folds from 0 to 9 are used for training
        urbansound_train = urbansound_metadata[urbansound_metadata.fold != 10]

        urbansound_train_filenames = self._get_filenames_by_class_id(urbansound_train)
        np.random.shuffle(urbansound_train_filenames)

        # separate noise files for train/validation
        urbansound_val = urbansound_train_filenames[-self.val_dataset_size:]
        urbansound_train = urbansound_train_filenames[:-self.val_dataset_size]
        print("Noise training:", len(urbansound_train))
        print("Noise validation:", len(urbansound_val))

        return urbansound_train, urbansound_val

    def get_test_filenames(self):
        urbansound_metadata = self._get_urban_sound_8K_filenames()

        # fold 10 is used for testing only
        urbansound_train = urbansound_metadata[urbansound_metadata.fold == 10]

        urbansound_test_filenames = self._get_filenames_by_class_id(urbansound_train)
        np.random.shuffle(urbansound_test_filenames)

        print("# of Noise testing files:", len(urbansound_test_filenames))
        return urbansound_test_filenames

class Dataset:
    def __init__(self, clean_filenames, noise_filenames, **config):
        self.clean_filenames = clean_filenames
        self.noise_filenames = noise_filenames
        self.sample_rate = config['fs']
        self.overlap = config['overlap']
        self.window_length = config['windowLength']
        self.audio_max_duration = config['audio_max_duration']

    def _sample_noise_filename(self):
        return np.random.choice(self.noise_filenames)

    def _remove_silent_frames(self, audio):
        trimed_audio = []
        indices = librosa.effects.split(audio, hop_length=self.overlap, top_db=20)

        for index in indices:
            trimed_audio.extend(audio[index[0]: index[1]])
        return np.array(trimed_audio)

    def _phase_aware_scaling(self, clean_spectral_magnitude, clean_phase, noise_phase):
        assert clean_phase.shape == noise_phase.shape, "Shapes must match."
        return clean_spectral_magnitude * np.cos(clean_phase - noise_phase)

    def get_noisy_audio(self, *, filename):
        return read_audio(filename, self.sample_rate)

    def _audio_random_crop(self, audio, duration):
        audio_duration_secs = librosa.core.get_duration(audio, self.sample_rate)

        ## duration: length of the cropped audio in seconds
        if duration >= audio_duration_secs:
            # print("Passed duration greater than audio duration of: ", audio_duration_secs)
            return audio

        audio_duration_ms = math.floor(audio_duration_secs * self.sample_rate)
        duration_ms = math.floor(duration * self.sample_rate)
        idx = np.random.randint(0, audio_duration_ms - duration_ms)
        return audio[idx: idx + duration_ms]

    def _add_noise_to_clean_audio(self, clean_audio, noise_signal):
        if len(clean_audio) >= len(noise_signal):
            # print("The noisy signal is smaller than the clean audio input. Duplicating the noise.")
            while len(clean_audio) >= len(noise_signal):
                noise_signal = np.append(noise_signal, noise_signal)

        ## Extract a noise segment from a random location in the noise file
        ind = np.random.randint(0, noise_signal.size - clean_audio.size)

        noiseSegment = noise_signal[ind: ind + clean_audio.size]

        speech_power = np.sum(clean_audio ** 2)
        noise_power = np.sum(noiseSegment ** 2)
        noisyAudio = clean_audio + np.sqrt(speech_power / noise_power) * noiseSegment
        return noisyAudio

    def parallel_audio_processing(self, clean_filename):
        
       
        clean_audio, _ = read_audio(clean_filename, self.sample_rate)

        # remove silent frame from clean audio
        clean_audio = self._remove_silent_frames(clean_audio)

        noise_filename = self._sample_noise_filename()

        # read the noise filename
        noise_audio, sr = read_audio(noise_filename, self.sample_rate)

        # remove silent frame from noise audio
        noise_audio = self._remove_silent_frames(noise_audio)

        # sample random fixed-sized snippets of audio
        clean_audio = self._audio_random_crop(clean_audio, duration=self.audio_max_duration)

        # add noise to input image
        noiseInput = self._add_noise_to_clean_audio(clean_audio, noise_audio)

        # extract stft features from noisy audio
        noisy_input_fe = FeatureExtractor(noiseInput, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        noise_spectrogram = noisy_input_fe.get_stft_spectrogram()

        # Or get the phase angle (in radians)
        # noisy_stft_magnitude, noisy_stft_phase = librosa.magphase(noisy_stft_features)
        noise_phase = np.angle(noise_spectrogram)

        # get the magnitude of the spectral
        noise_magnitude = np.abs(noise_spectrogram)

        # extract stft features from clean audio
        clean_audio_fe = FeatureExtractor(clean_audio, windowLength=self.window_length, overlap=self.overlap,
                                          sample_rate=self.sample_rate)
        clean_spectrogram = clean_audio_fe.get_stft_spectrogram()
        # clean_spectrogram = cleanAudioFE.get_mel_spectrogram()

        # get the clean phase
        clean_phase = np.angle(clean_spectrogram)

        # get the clean spectral magnitude
        clean_magnitude = np.abs(clean_spectrogram)
        # clean_magnitude = 2 * clean_magnitude / np.sum(scipy.signal.hamming(self.window_length, sym=False))

        clean_magnitude = self._phase_aware_scaling(clean_magnitude, clean_phase, noise_phase)

        scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        noise_magnitude = scaler.fit_transform(noise_magnitude)
        clean_magnitude = scaler.transform(clean_magnitude)

        return noise_magnitude, clean_magnitude, noise_phase

    def create_tf_record(self, *, prefix, subset_size, parallel=True):
        counter = 0
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        for i in range(0, len(self.clean_filenames), subset_size):
            
            tfrecord_filename = prefix + '_' + str(counter) + '.tfrecords'

            if os.path.isfile(tfrecord_filename):
                print(f"Skipping {tfrecord_filename}")
                counter += 1
                #continue

            writer = tf.io.TFRecordWriter(tfrecord_filename)
            clean_filenames_sublist = self.clean_filenames[i:i + subset_size]

            print(f"Processing files from: {i} to {i + subset_size}")
            print(*clean_filenames_sublist)
            
            if parallel:
                out = p.map(self.parallel_audio_processing, clean_filenames_sublist)
            else:
                out = [self.parallel_audio_processing(filename) for filename in clean_filenames_sublist]

            for o in out:
                noise_stft_magnitude = o[0]
                clean_stft_magnitude = o[1]
                noise_stft_phase = o[2]

                noise_stft_mag_features = prepare_input_features(noise_stft_magnitude, numSegments=8, numFeatures=129)

                noise_stft_mag_features = np.transpose(noise_stft_mag_features, (2, 0, 1))
                clean_stft_magnitude = np.transpose(clean_stft_magnitude, (1, 0))
                noise_stft_phase = np.transpose(noise_stft_phase, (1, 0))

                noise_stft_mag_features = np.expand_dims(noise_stft_mag_features, axis=3)
                clean_stft_magnitude = np.expand_dims(clean_stft_magnitude, axis=2)

                for x_, y_, p_ in zip(noise_stft_mag_features, clean_stft_magnitude, noise_stft_phase):
                    y_ = np.expand_dims(y_, 2)
                    example = get_tf_feature(x_, y_, p_)
                    writer.write(example.SerializeToString())

            counter += 1
            writer.close()

import warnings

warnings.filterwarnings(action='ignore')

mozilla_basepath = '../input/tamilvoiceclipsmozilla/'
urbansound_basepath = '../input/ultrasound/'

mcv = MozillaCommonVoiceDataset(mozilla_basepath, val_dataset_size=1000)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

us8K = UrbanSound8K(urbansound_basepath, val_dataset_size=200)
noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
val_dataset.create_tf_record(prefix='val', subset_size=2000)

train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
train_dataset.create_tf_record(prefix='train', subset_size=4000)

## Create Test Set
clean_test_filenames = mcv.get_test_filenames()

noise_test_filenames = us8K.get_test_filenames()

test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
test_dataset.create_tf_record(prefix='test', subset_size=1000, parallel=False)
