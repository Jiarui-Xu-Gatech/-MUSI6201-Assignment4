import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.io
from os.path import dirname, join as pjoin
import os

def get_spectral_peaks(X):
    pass

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    pass

def extract_pitch_chroma(X, fs, tfInHz):
    pass

def detect_key(x, blockSize, hopSize, fs, bTune):
    pass

def eval_tfe(pathToAudio, pathToGT):
    pass

def eval_key_detection(pathToAudio, pathToGT):
    pass

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    pass
