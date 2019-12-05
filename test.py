#File to initialize 2D image, pass input image to kernel, and peform timing analyis on output image.
#Authors. Kaylo Littlejohn and Desmond Yao 2019.

import numpy as np
from dwt_serial import *

"""
1. Test with some random array
"""
signal = np.random.rand(100, 100)

wav = gen_wavelet()
cA, cH, cV, cD = run_DWT(signal, wav, True, mode='zero')
rec_signal = run_iDWT(wav, cA, cH, cV, cD, mode='zero')

print('Rec same as original: {}'.format(np.allclose(signal, rec_signal)))