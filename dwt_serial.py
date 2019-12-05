import pywt
import numpy as np

signal = np.random.rand(100, 100)


def gen_wavelet():
    # Define the coefficients for the CDF9/7 filters
    factor=1

    # FORWARD FILTER COEFFICIENTS
    # Forward filter: lowpass
    cdf97_an_lo = factor * np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,
                                     0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443,
                                     0.026748757411])

    # Forward filter: highpass
    cdf97_an_hi = factor * np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,
                                     -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])

    # INVERSE FILTER COEFFICIENTS
    # Inverse filter: lowpass
    cdf97_syn_lo = factor * np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,
                                      0.591271763114, -0.057543526229, -0.091271763114, 0, 0])

    # Inverse filter: highpass
    cdf97_syn_hi = factor * np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,
                                      0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443,
                                      0.026748757411])

    cdf97 = pywt.Wavelet('cdf97', [cdf97_an_lo, cdf97_an_hi, cdf97_syn_lo, cdf97_syn_hi])

    wav = cdf97

    return wav

def run_DWT(wav, mode='zero'):
    a,d = pywt.dwt(signal,wav,mode)
    print("approx: %s \n detail: %s \n" %(a,d))

    return a, d


def run_iDWT(wav, a, d, mode):
    rec_sig = pywt.idwt(a,d,wav,mode)

    return rec_sig
