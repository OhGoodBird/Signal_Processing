#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

window_size = 256
window_shift = window_size // 2
hamming_window = np.hamming(window_size)
print('window size = {}'.format(window_size))
print('window shift = {}'.format(window_shift))

fs, xs = wavfile.read('./f1_Sen2_16000.wav')
len_xs = len(xs)

print('freq = {}'.format(fs))
print('len = {}'.format(len_xs))

spectrogram = []
num_frame = (len_xs - window_size) // window_shift + 1
print(num_frame)
for idx in range(num_frame):
    print('{}, {}, {}'.format(idx, idx*window_shift, idx*window_shift+window_size))
    xs_w = xs[idx*window_shift : idx*window_shift+window_size] * hamming_window
    
    xs_fft = np.fft.fft(xs_w)
    xs_fft = xs_fft[:window_size//2]
    xs_amp = np.absolute(xs_fft)
    xs_db = np.log(xs_amp)
    
    spectrogram.append(xs_db)

spectrogram = np.rot90(np.array(spectrogram))
imgplot = plt.imsave('spectrogram',spectrogram)
