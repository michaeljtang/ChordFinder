import numpy as np

def get_pitch(freq):
    """
    Takes in frequency in Hz, and outputs the closest note
    """
    # needed frequency dictionaries are preloaded for efficiency
    f_dict = {16.35: 'C', 17.32: 'C#/Db', 18.35: 'D', 19.45: 'D#/Eb', 20.6: 'E', 21.83: 'F', 23.12: 'F#/Gb', 24.5: 'G', 25.96: 'G#/Ab', 27.5: 'A', 29.14: 'A#/Bb', 30.87: 'B', 32.7: 'C', 34.65: 'C#/Db', 36.71: 'D', 38.89: 'D#/Eb', 41.2: 'E', 43.65: 'F', 46.25: 'F#/Gb', 49.0: 'G', 51.91: 'G#/Ab', 55.0: 'A', 58.27: 'A#/Bb', 61.74: 'B', 65.41: 'C', 69.3: 'C#/Db', 73.42: 'D', 77.78: 'D#/Eb', 82.41: 'E', 87.31: 'F', 92.5: 'F#/Gb', 98.0: 'G', 103.83: 'G#/Ab', 110.0: 'A', 116.54: 'A#/Bb', 123.47: 'B', 130.81: 'C', 138.59: 'C#/Db', 146.83: 'D', 155.56: 'D#/Eb', 164.81: 'E', 174.61: 'F', 185.0: 'F#/Gb', 196.0: 'G', 207.65: 'G#/Ab', 220.0: 'A', 233.08: 'A#/Bb', 246.94: 'B', 261.63: 'C', 277.18: 'C#/Db', 293.66: 'D', 311.13: 'D#/Eb', 329.63: 'E', 349.23: 'F', 369.99: 'F#/Gb', 392.0: 'G', 415.3: 'G#/Ab', 440.0: 'A', 466.16: 'A#/Bb', 493.88: 'B', 523.25: 'C', 554.37: 'C#/Db', 587.33: 'D', 622.25: 'D#/Eb', 659.25: 'E', 698.46: 'F', 739.99: 'F#/Gb', 783.99: 'G', 830.61: 'G#/Ab', 880.0: 'A', 932.33: 'A#/Bb', 987.77: 'B', 1046.5: 'C', 1108.73: 'C#/Db', 1174.66: 'D', 1244.51: 'D#/Eb', 1318.51: 'E', 1396.91: 'F', 1479.98: 'F#/Gb', 1567.98: 'G', 1661.22: 'G#/Ab', 1760.0: 'A', 1864.66: 'A#/Bb', 1975.53: 'B', 2093.0: 'C', 2217.46: 'C#/Db', 2349.32: 'D', 2489.02: 'D#/Eb', 2637.02: 'E', 2793.83: 'F', 2959.96: 'F#/Gb', 3135.96: 'G', 3322.44: 'G#/Ab', 3520.0: 'A', 3729.31: 'A#/Bb', 3951.07: 'B', 4186.01: 'C', 4434.92: 'C#/Db', 4698.63: 'D', 4978.03: 'D#/Eb', 5274.04: 'E', 5587.65: 'F', 5919.91: 'F#/Gb', 6271.93: 'G', 6644.88: 'G#/Ab', 7040.0: 'A', 7458.62: 'A#/Bb', 7902.13: 'B'}
    f_list = [16.35, 17.32, 18.35, 19.45, 20.6, 21.83, 23.12, 24.5, 25.96, 27.5, 29.14, 30.87, 32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49.0, 51.91, 55.0, 58.27, 61.74, 65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98.0, 103.83, 110.0, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.0, 196.0, 207.65, 220.0, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.0, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.0, 1864.66, 1975.53, 2093.0, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.0, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.0, 7458.62, 7902.13]
    closest_freq_idx = np.argmin(list(map(lambda x: abs(x - freq), f_list)))
    note = f_dict[f_list[closest_freq_idx]]
    return note

def get_chord(notes):
    """
    Takes in list of notes and returns the chord they correspond to
    """
