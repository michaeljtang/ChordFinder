import pyaudio
# stream allows us to easily decode bit data
import struct
import numpy as np
import matplotlib.pyplot as plt
from helper import *

def main():
    CHUNK = 1024 * 4 # number of samples per chunk
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 # audio channels
    RATE = 44100 # Hz

    # the main pyaudio object
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )

    # code from https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
    while True:
        data = stream.read(CHUNK)
        npdata = np.frombuffer(data, dtype=np.int16)
        # npdata needs to be converted to float so that we can calculate rms values
        npdata = npdata.astype('float32')
        chord = analyze_fft(npdata, 500, RATE)
        print(chord)

if __name__ == '__main__':
    main()