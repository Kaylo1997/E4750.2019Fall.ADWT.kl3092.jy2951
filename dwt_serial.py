import pywt
import numpy as np


def gen_wavelet():
    # Define the coefficients for the CDF9/7 filters
    factor=1

    # FORWARD FILTER COEFFICIENTS
    # Forward Decomposition filter: lowpass
    cdf97_an_lo = factor * np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,
                                     0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443,
                                     0.026748757411])

    # Forward Decomposition filter: highpass
    cdf97_an_hi = factor * np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,
                                     -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])

    # INVERSE FILTER COEFFICIENTS
    # Inverse Reconstruction filter: lowpass
    cdf97_syn_lo = factor * np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,
                                      0.591271763114, -0.057543526229, -0.091271763114, 0, 0])

    # Inverse Reconstruction filter: highpass
    cdf97_syn_hi = factor * np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,
                                      0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443,
                                      0.026748757411])

    cdf97 = pywt.Wavelet('cdf97', [cdf97_an_lo, cdf97_an_hi, cdf97_syn_lo, cdf97_syn_hi])

    wav = cdf97

    return wav


def run_DWT(signal, wav, flag_print, mode='zero'):
    coeffs = pywt.dwt2(signal, wav, mode)
    cA, (cH, cV, cD) = coeffs
    if flag_print:
        print("approx: {} \n detail: {} \n{}\n{}\n".format(cA, cH, cV, cD))

    return cA, cH, cV, cD


def run_iDWT(wav, cA, cH, cV, cD, mode):
    coeffs = cA, (cH, cV, cD)
    rec_sig = pywt.idwt2(coeffs, wav, mode)

    return rec_sig

