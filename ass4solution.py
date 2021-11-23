import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.io
from os.path import dirname, join as pjoin
import os

def get_spectral_peaks(X):
    top_k=20
    arr = X
    spectralPeaks=arr.argsort(axis=0)[::-1][0:top_k]
    return spectralPeaks

def  block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb,t)

def compute_spectrogram(xb, fs): 
    k = np.arange(0,xb.shape[0])
    hannWin = np.hanning(xb.shape[1])
    xb = xb*hannWin
    spectrums = np.abs(np.fft.fft(xb)[:,:xb.shape[1]//2+1])
    X=spectrums.T
    fInHz=np.fft.rfftfreq(X.shape[0], d=1./fs)
    return X,fInHz


def convert_freq2midi(fInHz, fA4InHz = 440):
    if type(fInHz)==np.ndarray:
        original_shape=fInHz.shape
        fInHz=np.squeeze(fInHz.reshape((1,-1)))
    def convert_freq2midi_scalar(f, fA4InHz):
 
        if f <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f/fA4InHz))
    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
       return convert_freq2midi_scalar(fInHz,fA4InHz)
    midi = np.zeros(fInHz.shape)
    for k,f in enumerate(fInHz):
        midi[k] =  convert_freq2midi_scalar(f,fA4InHz)

    midi=midi.reshape(original_shape)     
    return (midi)


def estimate_tuning_freq(x, blockSize, hopSize, fs):
    xb,t=block_audio(x,blockSize,hopSize,fs)
    X,fInHz=compute_spectrogram(xb, fs)
    spectralPeaks=get_spectral_peaks(X)*fs/(2*(X.shape[0]))
    pitch_frequency=np.power(2,(np.arange(1,128)-69)/12)*440
    pitch_corr=np.zeros(spectralPeaks.shape)
    for i in range(spectralPeaks.shape[0]):
        for j in range(spectralPeaks.shape[1]):
            pitch_corr[i,j]=np.abs(pitch_frequency-spectralPeaks[i,j]).argmin()+1
    freq_corr=np.power(2,(pitch_corr-69)/12)*440
    deviation=spectralPeaks-freq_corr
    deviation=deviation.flatten()
    histogram = np.histogram(deviation,bins=67)
    tInHz=440+deviation[np.argmax(histogram[0])]
    midi=convert_freq2midi(spectralPeaks,tInHz)*100
    truth=pitch_corr*100
    cent_deviation = midi.flatten() - truth.flatten()
    histogram = np.histogram(cent_deviation,bins=49)
    tuning_pitch=69+deviation[np.argmax(histogram[0])]/100
    return tInHz
    

def extract_pitch_chroma(X, fs, tInHz):
    f_C3=tInHz*(2**(-21/12))
    Octaves = 3
    Octave_pitch = 12

    coeff = np.zeros([Octave_pitch, X.shape[0]])
    for i in range(0,Octave_pitch):
        Look_freq=f_C3*(2**(i/Octave_pitch))
        #down half of the semi-tone and up half of the semi-tone would be the range we look
        OctaveBound=Look_freq* 2 * (X.shape[0] - 1) / fs*np.array([2**(-1 / (2 * Octave_pitch)), 2**(1 / (2 * Octave_pitch))])
        #corresponds to a range within 3 octaves
        for j in range(0,Octaves):
            actual_Bound=np.array([np.around(OctaveBound[0]*(2**j)).astype(int),np.around(OctaveBound[1]*(2**j)).astype(int)])
            coeff[i,actual_Bound[0]:actual_Bound[1]]=1/(actual_Bound[1]-actual_Bound[0])
    pitchChroma=np.dot(coeff, X**2)
    norm = np.sqrt((pitchChroma**2).sum(axis=0, keepdims=True))
    norm[norm == 0] = 1
    pitchChroma = pitchChroma / norm
    return pitchChroma

def detect_key(x, blockSize, hopSize, fs, bTune):
    xb,t=block_audio(x,blockSize,hopSize,fs)
    X,fInHz=compute_spectrogram(xb, fs)
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    if bTune==True:
        tInHz=estimate_tuning_freq(x, blockSize, hopSize, fs)
    else:
        tInHz=440
    pitchChroma=extract_pitch_chroma(X, fs, tInHz)
    t_pc[0]=t_pc[0]/np.sqrt(sum(t_pc[0]**2))
    t_pc[1]=t_pc[1]/np.sqrt(sum(t_pc[1]**2))
    distance=np.zeros(24)
    for i in range(12):
        new_t_pc_major=np.hstack((t_pc[0][12-i:12],t_pc[0][0:12-i]))/np.sqrt(np.sum(t_pc[0]**2))
        new_t_pc_minor=np.hstack((t_pc[1][12-i:12],t_pc[1][0:12-i]))/np.sqrt(np.sum(t_pc[1]**2))
        all_pitchChroma=np.sum(pitchChroma,axis=1)
        distance[i]=np.sqrt(np.sum((all_pitchChroma/np.sqrt(np.sum(all_pitchChroma))-new_t_pc_major)**2))
        distance[i+12]=np.sqrt(np.sum((all_pitchChroma/np.sqrt(np.sum(all_pitchChroma))-new_t_pc_minor)**2))
    key=distance.argmin()
    if key<=11 and key+3>11:
        key=(key+3)%12
    elif key+3>23:
        key=key+3-12
    else:
        key=key+3
    return key#because 0 represent A not C
        

def eval_tfe(pathToAudio, pathToGT):
    blockSize = 4096
    hopSize = 2048
    filenames=os.listdir(pathToAudio)
    audio=np.array([])
    GT=np.array([])
    for item in filenames:
        if item[-3:len(item)]=='wav':
            wav_fname=pathToAudio +'\\' +item
            fs, data = wavfile.read(wav_fname)
            tInHz=estimate_tuning_freq(data[:], blockSize, hopSize, fs)
            audio=np.append(audio,convert_freq2midi(tInHz, 440))
            txt_path=pathToGT+'\\'+item[0:-4]+'.txt'
            txt=open(txt_path,'r')
            GT_one=float(txt.readline())
            GT=np.append(GT,convert_freq2midi(GT_one, 440))
    return np.mean(np.abs(audio-GT))*100

            

            
def eval_key_detection(pathToAudio, pathToGT):
    blockSize = 4096
    hopSize = 2048
    filenames=os.listdir(pathToAudio)
    accuracy1=0
    accuracy2=0
    GT=0
    for item in filenames:
        if item[-3:len(item)]=='wav':
            wav_fname=pathToAudio +'\\' +item
            fs, data = wavfile.read(wav_fname)
            key1=detect_key(data[:], blockSize, hopSize, fs,True)
            key2=detect_key(data[:], blockSize, hopSize, fs,False)
            txt_path=pathToGT+'\\'+item[0:-4]+'.txt'
            txt=open(txt_path,'r')
            GT=int(txt.readline())
            #print(GT)
            #print(key1)
            #print(key2)
            if GT==key1:
                accuracy1+=1
            if GT==key2:
                accuracy2+=1
    return np.array([accuracy1/len(filenames),accuracy2/len(filenames)]).reshape(2,1)

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    avg_deviationInCent = eval_tfe(pathToAudioTf, pathToGTTf)
    print(avg_deviationInCent)
    avg_accuracy = eval_key_detection(pathToAudioKey, pathToGTKey)
    print(avg_accuracy)
    return avg_accuracy, avg_deviationInCent

evaluate('key_tf\\key_eval\\audio','key_tf\\key_eval\\GT','key_tf\\tuning_eval\\audio','key_tf\\tuning_eval\\GT')
