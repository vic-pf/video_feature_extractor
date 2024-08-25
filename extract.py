import torch as th
import math
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
import subprocess
import librosa
import scipy

# Audio processing functions


def extract_audio(input_file, output_file):
    """ Extracts audio at the native sampling rate into a separate wav file. """
    with open('ffmpeg_log.txt', 'w') as log_file:
        subprocess.call(['ffmpeg', '-i', input_file, '-vn',
                        output_file], stdout=log_file, stderr=log_file)


def stereo_to_mono_downsample(input_file, output_file, sample_rate=16000):
    """ Resamples wav file (we use 16 kHz). Convert from stereo to mono. Apply a gain of -4 to avoid clipping for mono to stereo conversion. """
    subprocess.call(['sox', input_file, output_file, 'gain',
                    '-4', 'channels', '1', 'rate', str(sample_rate)])


def LoadAudio(path, target_length=2048, use_raw_length=False):
    """ Convert audio wav file to mel spectrogram features """
    audio_type = 'melspectrogram'
    preemph_coef = 0.97
    sample_rate = 16000
    window_size = 0.025
    window_stride = 0.01
    window_type = 'hamming'
    num_mel_bins = 40
    padval = 0
    fmin = 20
    n_fft = int(sample_rate * window_size)
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)
    windows = {'hamming': scipy.signal.hamming}
    y, sr = librosa.load(path, sr=None)
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=windows[window_type])
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(
            sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        feats = librosa.power_to_db(melspec, ref=np.max)
    n_frames = feats.shape[1]
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        feats = np.pad(feats, ((0, 0), (0, p)), 'constant',
                       constant_values=(padval, padval))
    elif p < 0:
        feats = feats[:, 0:p]
    return feats, n_frames


def preemphasis(signal, coeff=0.97):
    """ Perform preemphasis on the input signal. """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# Argument parser
parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument('--csv', type=str, help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--type', type=str, default='2d', help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,
                    help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int,
                    default=4, help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                    help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str,
                    default='model/resnext101.pth', help='Resnext model path')
args = parser.parse_args()

# Video loader and model setup
dataset = VideoLoader(args.csv, framerate=1 if args.type == '2d' else 24,
                      size=224 if args.type == '2d' else 112, centercrop=(args.type == '3d'))
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(dataset, batch_size=1, shuffle=False,
                    num_workers=args.num_decoding_thread, sampler=sampler if n_dataset > 10 else None)
preprocess = Preprocessing(args.type)
model = get_model(args)

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data['input'][0]
        output_file_base = data['output'][0].replace('.npy', '')
        output_file_video = f"{output_file_base}/{args.type}.npy"
        output_file_audio = f"{output_file_base}/{args.type}.wav"
        output_file_audio_mono = output_file_audio.replace('.wav', '_mono.wav')
        output_file_audio_features = output_file_audio.replace('.wav', '.npy')

        if len(data['video'].shape) > 3:
            print('Computing features of video {}/{}: {}'.format(k +
                  1, n_dataset, input_file))
            video = data['video'].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype('float16')
                np.save(output_file_video, features)

                # Extract audio
                extract_audio(input_file, output_file_audio)
                stereo_to_mono_downsample(
                    output_file_audio, output_file_audio_mono)

                # Load and save audio features
                audio_features, _ = LoadAudio(output_file_audio_mono)
                np.save(output_file_audio_features, audio_features)
        else:
            print('Video {} already processed.'.format(input_file))
