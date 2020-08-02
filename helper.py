import numpy as np
from mingus.core.chords import determine
import functools
from scipy import fft

def get_pitch(freq):
    """
    Takes in frequency in Hz, and outputs the closest note
    """
    # needed frequency dictionaries are preloaded for efficiency
#    f_dict = {16.35: 'C', 17.32: 'C#/Db', 18.35: 'D', 19.45: 'D#/Eb', 20.6: 'E', 21.83: 'F', 23.12: 'F#/Gb', 24.5: 'G', 25.96: 'G#/Ab', 27.5: 'A', 29.14: 'A#/Bb', 30.87: 'B', 32.7: 'C', 34.65: 'C#/Db', 36.71: 'D', 38.89: 'D#/Eb', 41.2: 'E', 43.65: 'F', 46.25: 'F#/Gb', 49.0: 'G', 51.91: 'G#/Ab', 55.0: 'A', 58.27: 'A#/Bb', 61.74: 'B', 65.41: 'C', 69.3: 'C#/Db', 73.42: 'D', 77.78: 'D#/Eb', 82.41: 'E', 87.31: 'F', 92.5: 'F#/Gb', 98.0: 'G', 103.83: 'G#/Ab', 110.0: 'A', 116.54: 'A#/Bb', 123.47: 'B', 130.81: 'C', 138.59: 'C#/Db', 146.83: 'D', 155.56: 'D#/Eb', 164.81: 'E', 174.61: 'F', 185.0: 'F#/Gb', 196.0: 'G', 207.65: 'G#/Ab', 220.0: 'A', 233.08: 'A#/Bb', 246.94: 'B', 261.63: 'C', 277.18: 'C#/Db', 293.66: 'D', 311.13: 'D#/Eb', 329.63: 'E', 349.23: 'F', 369.99: 'F#/Gb', 392.0: 'G', 415.3: 'G#/Ab', 440.0: 'A', 466.16: 'A#/Bb', 493.88: 'B', 523.25: 'C', 554.37: 'C#/Db', 587.33: 'D', 622.25: 'D#/Eb', 659.25: 'E', 698.46: 'F', 739.99: 'F#/Gb', 783.99: 'G', 830.61: 'G#/Ab', 880.0: 'A', 932.33: 'A#/Bb', 987.77: 'B', 1046.5: 'C', 1108.73: 'C#/Db', 1174.66: 'D', 1244.51: 'D#/Eb', 1318.51: 'E', 1396.91: 'F', 1479.98: 'F#/Gb', 1567.98: 'G', 1661.22: 'G#/Ab', 1760.0: 'A', 1864.66: 'A#/Bb', 1975.53: 'B', 2093.0: 'C', 2217.46: 'C#/Db', 2349.32: 'D', 2489.02: 'D#/Eb', 2637.02: 'E', 2793.83: 'F', 2959.96: 'F#/Gb', 3135.96: 'G', 3322.44: 'G#/Ab', 3520.0: 'A', 3729.31: 'A#/Bb', 3951.07: 'B', 4186.01: 'C', 4434.92: 'C#/Db', 4698.63: 'D', 4978.03: 'D#/Eb', 5274.04: 'E', 5587.65: 'F', 5919.91: 'F#/Gb', 6271.93: 'G', 6644.88: 'G#/Ab', 7040.0: 'A', 7458.62: 'A#/Bb', 7902.13: 'B'}
    f_dict = {16.35: 'C0', 17.32: 'C#0/Db0', 18.35: 'D0', 19.45: 'D#0/Eb0', 20.6: 'E0', 21.83: 'F0', 23.12: 'F#0/Gb0', 24.5: 'G0', 25.96: 'G#0/Ab0', 27.5: 'A0', 29.14: 'A#0/Bb0', 30.87: 'B0', 32.7: 'C1', 34.65: 'C#1/Db1', 36.71: 'D1', 38.89: 'D#1/Eb1', 41.2: 'E1', 43.65: 'F1', 46.25: 'F#1/Gb1', 49.0: 'G1', 51.91: 'G#1/Ab1', 55.0: 'A1', 58.27: 'A#1/Bb1', 61.74: 'B1', 65.41: 'C2', 69.3: 'C#2/Db2', 73.42: 'D2', 77.78: 'D#2/Eb2', 82.41: 'E2', 87.31: 'F2', 92.5: 'F#2/Gb2', 98.0: 'G2', 103.83: 'G#2/Ab2', 110.0: 'A2', 116.54: 'A#2/Bb2', 123.47: 'B2', 130.81: 'C3', 138.59: 'C#3/Db3', 146.83: 'D3', 155.56: 'D#3/Eb3', 164.81: 'E3', 174.61: 'F3', 185.0: 'F#3/Gb3', 196.0: 'G3', 207.65: 'G#3/Ab3', 220.0: 'A3', 233.08: 'A#3/Bb3', 246.94: 'B3', 261.63: 'C4', 277.18: 'C#4/Db4', 293.66: 'D4', 311.13: 'D#4/Eb4', 329.63: 'E4', 349.23: 'F4', 369.99: 'F#4/Gb4', 392.0: 'G4', 415.3: 'G#4/Ab4', 440.0: 'A4', 466.16: 'A#4/Bb4', 493.88: 'B4', 523.25: 'C5', 554.37: 'C#5/Db5', 587.33: 'D5', 622.25: 'D#5/Eb5', 659.25: 'E5', 698.46: 'F5', 739.99: 'F#5/Gb5', 783.99: 'G5', 830.61: 'G#5/Ab5', 880.0: 'A5', 932.33: 'A#5/Bb5', 987.77: 'B5', 1046.5: 'C6', 1108.73: 'C#6/Db6', 1174.66: 'D6', 1244.51: 'D#6/Eb6', 1318.51: 'E6', 1396.91: 'F6', 1479.98: 'F#6/Gb6', 1567.98: 'G6', 1661.22: 'G#6/Ab6', 1760.0: 'A6', 1864.66: 'A#6/Bb6', 1975.53: 'B6', 2093.0: 'C7', 2217.46: 'C#7/Db7', 2349.32: 'D7', 2489.02: 'D#7/Eb7', 2637.02: 'E7', 2793.83: 'F7', 2959.96: 'F#7/Gb7', 3135.96: 'G7', 3322.44: 'G#7/Ab7', 3520.0: 'A7', 3729.31: 'A#7/Bb7', 3951.07: 'B7', 4186.01: 'C8', 4434.92: 'C#8/Db8', 4698.63: 'D8', 4978.03: 'D#8/Eb8', 5274.04: 'E8', 5587.65: 'F8', 5919.91: 'F#8/Gb8', 6271.93: 'G8', 6644.88: 'G#8/Ab8', 7040.0: 'A8', 7458.62: 'A#8/Bb8', 7902.13: 'B8'}
    f_list = [16.35, 17.32, 18.35, 19.45, 20.6, 21.83, 23.12, 24.5, 25.96, 27.5, 29.14, 30.87, 32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49.0, 51.91, 55.0, 58.27, 61.74, 65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98.0, 103.83, 110.0, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.0, 196.0, 207.65, 220.0, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.0, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.0, 1864.66, 1975.53, 2093.0, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.0, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.0, 7458.62, 7902.13]
    closest_freq_idx = np.argmin(list(map(lambda x: abs(x - freq), f_list)))
    note = f_dict[f_list[closest_freq_idx]]
    return note

def rms(wf):
    """
    Calculates rms value of waveform wf
    """
    return np.sqrt(np.mean(wf ** 2))

def to_histogram(data):
    """
    takes data (type: list) and converts to a histogram (type: dict)
    """
    hist = {}
    for element in data:
        if element in hist.keys():
            hist[element] += 1
        else:
            hist[element] = 1
    return hist

def strip_octave(note):
    """
    Takes in note in octave format (ie. C7, A5, etc.) and strips off the number, only
    leaving the note name.

    For enharmonic groups (ie. 'C#7/Db7), only returns the sharp version with octave stripped
    """
    if note[1].isdigit():
        return note[0]
    else:
        return note[:2]

def extract_chord(func):
    """
    func is the waveform processing method, and then this wrapper determines what happens
    after a list of potential notes has been determined
    """
    @functools.wraps(func)
    def wrapper_extract_chord(wf, threshold, sample_rate):
        # verify threshold criteria
        rms_value = rms(wf)
        if rms_value < threshold:
            return []
        
        # list of note frequencies detected by whatever algorithm 'func' was chosen
        potential_notes = func(wf, threshold, sample_rate) 

        # process notes
        note_histo = to_histogram(potential_notes)
        note_histo_keys = list(note_histo.keys())
        note_histo_keys.sort(reverse=True, key=lambda x : note_histo[x])
        
        # most basic -- just take top 3
        top_notes = note_histo_keys[:3]
        print(top_notes)
        
        # remove octave labels
        notes = list(map(strip_octave, top_notes))

        return determine(notes)

    return wrapper_extract_chord

############################# a couple of different functions that can be used to analyze note data

@extract_chord
def analyze_fft(wf, threshold, sample_rate):
    """
    Input:
    wf - waveform of signal to analyze
    threshold - minimum rms value the waveform needs to have to be recognized as an input

    Output:
    list of notes in chord contained in waveform
    """
    samples = len(wf)

    # fourier transform
    fhat = fft.fft(wf)
    # think this is actually the square root of the PSD? not sure what else to call it
    PSD = np.abs(fhat)
    freq = fft.fftfreq(samples, 1 / sample_rate)
    pos_freq = freq[:]
    pos_PSD = PSD[freq >= 0]

    # # find max
    # print(pos_freq[np.argmax(pos_PSD)])

    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    max_freqs = pos_freq[pos_PSD.argsort()[-100:][::-1]]
    return list(map(get_pitch, max_freqs))

# print(determine(['F', 'B', 'A#']))
