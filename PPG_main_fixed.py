"""
-+-+-+ IMPORTANT NOTES (READ FIRST) +-+-+-
This is the refactored backend logic module (PPG_main_fixed.py). It contains
all the signal processing classes and functions but NO PLOTTING code. It is
designed to be imported by a separate GUI file.

--- MODIFICATION HIGHLIGHTS ---
- [PLOT FIX] Corrected the `calculate_qj_frequency_responses` function to properly
  scale the frequency axis. It now uses a single high-resolution template for Hw/Gw
  and samples from it, preserving the plot's morphology for downsampled signals.
- [AMPLITUDE FIX] Added a signal normalization step in `HRV_Analyzer._preprocess_and_filter`.
  The output signal is now scaled to a standard deviation of 1, making its amplitude
  consistent regardless of the downsampling factor.
- [HRV FIX] The amplitude normalization also fixes the bug where pNN50 was always zero
  by providing a stable signal to the peak detector, allowing it to capture true
  beat-to-beat variability.

Thanks,
Jemi
5023231017

GG
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# --- MANUAL IMPLEMENTATIONS (REPLACING SCIPY) ---

def manual_butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low_w = 2 * fs * np.tan(np.pi * low / fs)
    high_w = 2 * fs * np.tan(np.pi * high / fs)
    poles = np.exp(1j * np.pi * (2 * np.arange(1, order + 1) + order - 1) / (2 * order))
    alpha = np.cos(np.pi * (high + low) / (2 * fs)) / np.cos(np.pi * (high - low) / (2 * fs))
    k = np.tan(np.pi * (high - low) / (2 * fs))
    p_bp = []
    for p in poles:
        radical = np.sqrt(k**2 * p**2 - 4 * alpha**2 + 0j)
        p_bp.append((k * p + radical) / (2 * alpha))
        p_bp.append((k * p - radical) / (2 * alpha))
    z_p = [(2 + p) / (2 - p) for p in p_bp]
    z_z = [1, -1] * order
    a = np.poly(z_p)
    b = np.poly(z_z)
    center_freq_rad = np.pi * (high + low) / 2
    w_eval = np.exp(1j * center_freq_rad)
    gain = np.abs(np.polyval(b, w_eval) / np.polyval(a, w_eval))
    if gain != 0: b /= gain
    return b.real, a.real

def manual_lfilter(b, a, x):
    y = np.zeros_like(x)
    for n in range(len(x)):
        forward_sum = 0
        for k in range(len(b)):
            if n - k >= 0: forward_sum += b[k] * x[n - k]
        feedback_sum = 0
        for k in range(1, len(a)):
            if n - k >= 0: feedback_sum += a[k] * y[n - k]
        y[n] = (forward_sum - feedback_sum) / a[0]
    return y

def manual_find_peaks(x, prominence=None, distance=None, height=None):
    dx = np.diff(x)
    peaks_idx = np.where((np.hstack([dx, 0]) <= 0) & (np.hstack([0, dx]) > 0))[0]
    if height is not None:
        peaks_idx = peaks_idx[x[peaks_idx] >= height]
    if len(peaks_idx) == 0: return np.array([], dtype=int)
    if distance is not None:
        peaks_idx = peaks_idx[np.argsort(-x[peaks_idx])]
        keep = np.ones(len(peaks_idx), dtype=bool)
        for i in range(len(peaks_idx)):
            if not keep[i]: continue
            current_peak_loc = peaks_idx[i]
            for j in range(i + 1, len(peaks_idx)):
                if abs(current_peak_loc - peaks_idx[j]) < distance:
                    keep[j] = False
        peaks_idx = peaks_idx[keep]
        peaks_idx = np.sort(peaks_idx)
    if prominence is not None and len(peaks_idx) > 0:
        prominences = []
        for peak in peaks_idx:
            left_base_idx = 0
            for i in range(peak - 1, -1, -1):
                if x[i] > x[peak]: left_base_idx = i; break
            right_base_idx = len(x) - 1
            for i in range(peak + 1, len(x)):
                if x[i] > x[peak]: right_base_idx = i; break
            lowest_contour = np.min(x[left_base_idx:right_base_idx+1]) if left_base_idx < right_base_idx else x[peak]
            prominences.append(x[peak] - lowest_contour)
        peaks_idx = peaks_idx[np.array(prominences) >= prominence]
    return peaks_idx

# --- UTILITY FUNCTIONS ---

def welch_from_scratch(signal_data, fs, segment_len=256, overlap_ratio=0.5, fft_func=None):
    x = np.asarray(signal_data, dtype=float)
    if len(x) < 4: return np.array([]), np.array([])
    nperseg = min(segment_len, len(x))
    if nperseg < 4: nperseg = len(x)
    noverlap = int(nperseg * overlap_ratio)
    step = nperseg - noverlap
    if step <= 0: step = 1
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    num_segments = (len(x) - noverlap) // step
    psd_segments = []
    if fft_func is None: raise ValueError("fft_func must be supplied to welch_from_scratch.")
    for i in range(num_segments):
        start = i * step
        end = start + nperseg
        if end > len(x): break
        segment = x[start:end] * window
        fft_complex = fft_func(segment)
        power_spectrum = (np.abs(fft_complex)**2) / (fs * np.sum(window**2))
        psd_segments.append(power_spectrum)
    if not psd_segments: return np.array([]), np.array([])
    avg_psd = np.mean(psd_segments, axis=0)
    N_fft = len(avg_psd)
    frequencies = np.fft.fftfreq(N_fft, 1.0/fs)
    half = N_fft // 2
    psd_single_sided = avg_psd[:half] * 2
    if len(psd_single_sided) > 0: psd_single_sided[0] /= 2
    return frequencies[:half], psd_single_sided

def extract_rate_from_signal(signal, fs, freq_band, fft_func):
    if signal is None or len(signal) < 10 or fs <= 0: return 0
    freqs, psd = welch_from_scratch(signal - np.mean(signal), fs, segment_len=min(512, len(signal)), fft_func=fft_func)
    if freqs.size == 0 or psd.size == 0: return 0
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if not np.any(band_mask): return 0
    peak_freq_in_band = freqs[band_mask][np.argmax(psd[band_mask])]
    return peak_freq_in_band * 60

# --- ANALYSIS CLASSES ---

class PPGStressAnalyzer:
    def __init__(self, sampling_rate=125.0):
        self.fs = sampling_rate
        self.original_fs = sampling_rate
        self.qj_time_coeffs = self._initialize_qj_time_coeffs()
        
    def dirac_delta(self, x):
        return 1 if x == 0 else 0

    def _initialize_qj_time_coeffs(self):
        qj_filters = {}
        for j in range(1, 9):
            filter_dict = {}
            start_k = -(2**j + 2**(j-1) - 2)
            end_k = (1 - 2**(j-1)) + 1
            if j == 1:
                for k in range(start_k, end_k): filter_dict[k] = -2 * (self.dirac_delta(k) - self.dirac_delta(k + 1))
            elif j == 2:
                for k in range(start_k, end_k): filter_dict[k] = -1/4 * (self.dirac_delta(k-1) + 3*self.dirac_delta(k) + 2*self.dirac_delta(k+1) - 2*self.dirac_delta(k+2) - 3*self.dirac_delta(k+3) - self.dirac_delta(k+4))
            elif j == 3:
                for k in range(start_k, end_k): filter_dict[k] = -1/32 * (self.dirac_delta(k-3) + 3*self.dirac_delta(k-2) + 6*self.dirac_delta(k-1) + 10*self.dirac_delta(k) + 11*self.dirac_delta(k+1) + 9*self.dirac_delta(k+2) + 4*self.dirac_delta(k+3) - 4*self.dirac_delta(k+4) - 9*self.dirac_delta(k+5) - 11*self.dirac_delta(k+6) - 10*self.dirac_delta(k+7) - 6*self.dirac_delta(k+8) - 3*self.dirac_delta(k+9) - self.dirac_delta(k+10))
            elif j == 4:
                for k in range(start_k, end_k): filter_dict[k] = -1/256 * (self.dirac_delta(k-7) + 3*self.dirac_delta(k-6) + 6*self.dirac_delta(k-5) + 10*self.dirac_delta(k-4) + 15*self.dirac_delta(k-3) + 21*self.dirac_delta(k-2) + 28*self.dirac_delta(k-1) + 36*self.dirac_delta(k) + 41*self.dirac_delta(k+1) + 43*self.dirac_delta(k+2) + 42*self.dirac_delta(k+3) + 38*self.dirac_delta(k+4) + 31*self.dirac_delta(k+5) + 21*self.dirac_delta(k+6) + 8*self.dirac_delta(k+7) - 8*self.dirac_delta(k+8) - 21*self.dirac_delta(k+9) - 31*self.dirac_delta(k+10) - 38*self.dirac_delta(k+11) - 42*self.dirac_delta(k+12) - 43*self.dirac_delta(k+13) - 41*self.dirac_delta(k+14) - 36*self.dirac_delta(k+15) - 28*self.dirac_delta(k+16) - 21*self.dirac_delta(k+17) - 15*self.dirac_delta(k+18) - 10*self.dirac_delta(k+19) - 6*self.dirac_delta(k+20) - 3*self.dirac_delta(k+21) - self.dirac_delta(k+22))
            elif j == 5:
                 for k in range(start_k, end_k): filter_dict[k] = -1/2048 * (self.dirac_delta(k-15) + 3*self.dirac_delta(k-14) + 6*self.dirac_delta(k-13) + 10*self.dirac_delta(k-12) + 15*self.dirac_delta(k-11) + 21*self.dirac_delta(k-10) + 28*self.dirac_delta(k-9) + 36*self.dirac_delta(k-8) + 45*self.dirac_delta(k-7) + 55*self.dirac_delta(k-6) + 66*self.dirac_delta(k-5) + 78*self.dirac_delta(k-4) + 91*self.dirac_delta(k-3) + 105*self.dirac_delta(k-2) + 120*self.dirac_delta(k-1) + 136*self.dirac_delta(k) + 149*self.dirac_delta(k+1) + 159*self.dirac_delta(k+2) + 166*self.dirac_delta(k+3) + 170*self.dirac_delta(k+4) + 171*self.dirac_delta(k+5) + 169*self.dirac_delta(k+6) + 164*self.dirac_delta(k+7) + 156*self.dirac_delta(k+8) + 145*self.dirac_delta(k+9) + 131*self.dirac_delta(k+10) + 114*self.dirac_delta(k+11) + 94*self.dirac_delta(k+12) + 71*self.dirac_delta(k+13) + 45*self.dirac_delta(k+14) + 16*self.dirac_delta(k+15) - 16*self.dirac_delta(k+16) - 45*self.dirac_delta(k+17) - 71*self.dirac_delta(k+18) - 94*self.dirac_delta(k+19) - 114*self.dirac_delta(k+20) - 131*self.dirac_delta(k+21) - 145*self.dirac_delta(k+22) - 156*self.dirac_delta(k+23) - 164*self.dirac_delta(k+24) - 169*self.dirac_delta(k+25) - 171*self.dirac_delta(k+26) - 170*self.dirac_delta(k+27) - 166*self.dirac_delta(k+28) - 159*self.dirac_delta(k+29) - 149*self.dirac_delta(k+30) - 136*self.dirac_delta(k+31) - 120*self.dirac_delta(k+32) - 105*self.dirac_delta(k+33) - 91*self.dirac_delta(k+34) - 78*self.dirac_delta(k+35) - 66*self.dirac_delta(k+36) - 55*self.dirac_delta(k+37) - 45*self.dirac_delta(k+38) - 36*self.dirac_delta(k+39) - 28*self.dirac_delta(k+40) - 21*self.dirac_delta(k+41) - 15*self.dirac_delta(k+42) - 10*self.dirac_delta(k+43) - 6*self.dirac_delta(k+44) - 3*self.dirac_delta(k+45) - self.dirac_delta(k+46))
            elif j == 6:
                for k in range(start_k, end_k): filter_dict[k] = -1/16384 * (self.dirac_delta(k-31) + 3*self.dirac_delta(k-30) + 6*self.dirac_delta(k-29) + 10*self.dirac_delta(k-28) + 15*self.dirac_delta(k-27) + 21*self.dirac_delta(k-26) + 28*self.dirac_delta(k-25) + 36*self.dirac_delta(k-24) + 45*self.dirac_delta(k-23) + 55*self.dirac_delta(k-22) + 66*self.dirac_delta(k-21) + 78*self.dirac_delta(k-20) + 91*self.dirac_delta(k-19) + 105*self.dirac_delta(k-18) + 120*self.dirac_delta(k-17) + 136*self.dirac_delta(k-16) + 153*self.dirac_delta(k-15) + 171*self.dirac_delta(k-14) + 190*self.dirac_delta(k-13) + 210*self.dirac_delta(k-12) + 231*self.dirac_delta(k-11) + 253*self.dirac_delta(k-10) + 276*self.dirac_delta(k-9) + 300*self.dirac_delta(k-8) + 325*self.dirac_delta(k-7) + 351*self.dirac_delta(k-6) + 378*self.dirac_delta(k-5) + 406*self.dirac_delta(k-4) + 435*self.dirac_delta(k-3) + 465*self.dirac_delta(k-2) + 496*self.dirac_delta(k-1) + 528*self.dirac_delta(k) + 557*self.dirac_delta(k+1) + 583*self.dirac_delta(k+2) + 606*self.dirac_delta(k+3) + 626*self.dirac_delta(k+4) + 643*self.dirac_delta(k+5) + 657*self.dirac_delta(k+6) + 668*self.dirac_delta(k+7) + 676*self.dirac_delta(k+8) + 681*self.dirac_delta(k+9) + 683*self.dirac_delta(k+10) + 682*self.dirac_delta(k+11) + 678*self.dirac_delta(k+12) + 671*self.dirac_delta(k+13) + 661*self.dirac_delta(k+14) + 648*self.dirac_delta(k+15) + 632*self.dirac_delta(k+16) + 613*self.dirac_delta(k+17) + 591*self.dirac_delta(k+18) + 566*self.dirac_delta(k+19) + 538*self.dirac_delta(k+20) + 507*self.dirac_delta(k+21) + 473*self.dirac_delta(k+22) + 436*self.dirac_delta(k+23) + 396*self.dirac_delta(k+24) + 353*self.dirac_delta(k+25) + 307*self.dirac_delta(k+26) + 258*self.dirac_delta(k+27) + 206*self.dirac_delta(k+28) + 151*self.dirac_delta(k+29) + 93*self.dirac_delta(k+30) + 32*self.dirac_delta(k+31) - 32*self.dirac_delta(k+32) - 93*self.dirac_delta(k+33) - 151*self.dirac_delta(k+34) - 206*self.dirac_delta(k+35) - 258*self.dirac_delta(k+36) - 307*self.dirac_delta(k+37) - 353*self.dirac_delta(k+38) - 396*self.dirac_delta(k+39) - 436*self.dirac_delta(k+40) - 473*self.dirac_delta(k+41) - 507*self.dirac_delta(k+42) - 538*self.dirac_delta(k+43) - 566*self.dirac_delta(k+44) - 591*self.dirac_delta(k+45) - 613*self.dirac_delta(k+46) - 632*self.dirac_delta(k+47) - 648*self.dirac_delta(k+48) - 661*self.dirac_delta(k+49) - 671*self.dirac_delta(k+50) - 678*self.dirac_delta(k+51) - 682*self.dirac_delta(k+52) - 683*self.dirac_delta(k+53) - 681*self.dirac_delta(k+54) - 676*self.dirac_delta(k+55) - 668*self.dirac_delta(k+56) - 657*self.dirac_delta(k+57) - 643*self.dirac_delta(k+58) - 626*self.dirac_delta(k+59) - 606*self.dirac_delta(k+60) - 583*self.dirac_delta(k+61) - 557*self.dirac_delta(k+62) - 528*self.dirac_delta(k+63) - 496*self.dirac_delta(k+64) - 465*self.dirac_delta(k+65) - 435*self.dirac_delta(k+66) - 406*self.dirac_delta(k+67) - 378*self.dirac_delta(k+68) - 351*self.dirac_delta(k+69) - 325*self.dirac_delta(k+70) - 300*self.dirac_delta(k+71) - 276*self.dirac_delta(k+72) - 253*self.dirac_delta(k+73) - 231*self.dirac_delta(k+74) - 210*self.dirac_delta(k+75) - 190*self.dirac_delta(k+76) - 171*self.dirac_delta(k+77) - 153*self.dirac_delta(k+78) - 136*self.dirac_delta(k+79) - 120*self.dirac_delta(k+80) - 105*self.dirac_delta(k+81) - 91*self.dirac_delta(k+82) - 78*self.dirac_delta(k+83) - 66*self.dirac_delta(k+84) - 55*self.dirac_delta(k+85) - 45*self.dirac_delta(k+86) - 36*self.dirac_delta(k+87) - 28*self.dirac_delta(k+88) - 21*self.dirac_delta(k+89) - 15*self.dirac_delta(k+90) - 10*self.dirac_delta(k+91) - 6*self.dirac_delta(k+92) - 3*self.dirac_delta(k+93) - self.dirac_delta(k+94))
            elif j == 7:
                for k in range(start_k, end_k): filter_dict[k] = -1/131072 * (self.dirac_delta(k-63) + 3*self.dirac_delta(k-62) + 6*self.dirac_delta(k-61) + 10*self.dirac_delta(k-60) + 15*self.dirac_delta(k-59) + 21*self.dirac_delta(k-58) + 28*self.dirac_delta(k-57) + 36*self.dirac_delta(k-56) + 45*self.dirac_delta(k-55) + 55*self.dirac_delta(k-54) + 66*self.dirac_delta(k-53) + 78*self.dirac_delta(k-52) + 91*self.dirac_delta(k-51) + 105*self.dirac_delta(k-50) + 120*self.dirac_delta(k-49) + 136*self.dirac_delta(k-48) + 153*self.dirac_delta(k-47) + 171*self.dirac_delta(k-46) + 190*self.dirac_delta(k-45) + 210*self.dirac_delta(k-44) + 231*self.dirac_delta(k-43) + 253*self.dirac_delta(k-42) + 276*self.dirac_delta(k-41) + 300*self.dirac_delta(k-40) + 325*self.dirac_delta(k-39) + 351*self.dirac_delta(k-38) + 378*self.dirac_delta(k-37) + 406*self.dirac_delta(k-36) + 435*self.dirac_delta(k-35) + 465*self.dirac_delta(k-34) + 496*self.dirac_delta(k-33) + 528*self.dirac_delta(k-32) + 561*self.dirac_delta(k-31) + 595*self.dirac_delta(k-30) + 630*self.dirac_delta(k-29) + 666*self.dirac_delta(k-28) + 703*self.dirac_delta(k-27) + 741*self.dirac_delta(k-26) + 780*self.dirac_delta(k-25) + 820*self.dirac_delta(k-24) + 861*self.dirac_delta(k-23) + 903*self.dirac_delta(k-22) + 946*self.dirac_delta(k-21) + 990*self.dirac_delta(k-20) + 1035*self.dirac_delta(k-19) + 1081*self.dirac_delta(k-18) + 1128*self.dirac_delta(k-17) + 1176*self.dirac_delta(k-16) + 1225*self.dirac_delta(k-15) + 1275*self.dirac_delta(k-14) + 1326*self.dirac_delta(k-13) + 1378*self.dirac_delta(k-12) + 1431*self.dirac_delta(k-11) + 1485*self.dirac_delta(k-10) + 1540*self.dirac_delta(k-9) + 1596*self.dirac_delta(k-8) + 1653*self.dirac_delta(k-7) + 1711*self.dirac_delta(k-6) + 1770*self.dirac_delta(k-5) + 1830*self.dirac_delta(k-4) + 1891*self.dirac_delta(k-3) + 1953*self.dirac_delta(k-2) + 2016*self.dirac_delta(k-1) + 2080*self.dirac_delta(k) + 2141*self.dirac_delta(k+1) + 2199*self.dirac_delta(k+2) + 2254*self.dirac_delta(k+3) + 2306*self.dirac_delta(k+4) + 2355*self.dirac_delta(k+5) + 2401*self.dirac_delta(k+6) + 2444*self.dirac_delta(k+7) + 2484*self.dirac_delta(k+8) + 2521*self.dirac_delta(k+9) + 2555*self.dirac_delta(k+10) + 2586*self.dirac_delta(k+11) + 2614*self.dirac_delta(k+12) + 2639*self.dirac_delta(k+13) + 2661*self.dirac_delta(k+14) + 2680*self.dirac_delta(k+15) + 2696*self.dirac_delta(k+16) + 2709*self.dirac_delta(k+17) + 2719*self.dirac_delta(k+18) + 2726*self.dirac_delta(k+19) + 2730*self.dirac_delta(k+20) + 2731*self.dirac_delta(k+21) + 2729*self.dirac_delta(k+22) + 2724*self.dirac_delta(k+23) + 2716*self.dirac_delta(k+24) + 2705*self.dirac_delta(k+25) + 2691*self.dirac_delta(k+26) + 2674*self.dirac_delta(k+27) + 2654*self.dirac_delta(k+28) + 2631*self.dirac_delta(k+29) + 2605*self.dirac_delta(k+30) + 2576*self.dirac_delta(k+31) + 2544*self.dirac_delta(k+32) + 2509*self.dirac_delta(k+33) + 2471*self.dirac_delta(k+34) + 2430*self.dirac_delta(k+35) + 2386*self.dirac_delta(k+36) + 2339*self.dirac_delta(k+37) + 2289*self.dirac_delta(k+38) + 2236*self.dirac_delta(k+39) + 2180*self.dirac_delta(k+40) + 2121*self.dirac_delta(k+41) + 2059*self.dirac_delta(k+42) + 1994*self.dirac_delta(k+43) + 1926*self.dirac_delta(k+44) + 1855*self.dirac_delta(k+45) + 1781*self.dirac_delta(k+46) + 1704*self.dirac_delta(k+47) + 1624*self.dirac_delta(k+48) + 1541*self.dirac_delta(k+49) + 1455*self.dirac_delta(k+50) + 1366*self.dirac_delta(k+51) + 1274*self.dirac_delta(k+52) + 1179*self.dirac_delta(k+53) + 1081*self.dirac_delta(k+54) + 980*self.dirac_delta(k+55) + 876*self.dirac_delta(k+56) + 769*self.dirac_delta(k+57) + 659*self.dirac_delta(k+58) + 546*self.dirac_delta(k+59) + 430*self.dirac_delta(k+60) + 311*self.dirac_delta(k+61) + 189*self.dirac_delta(k+62) + 64*self.dirac_delta(k+63) - 64*self.dirac_delta(k+64) - 189*self.dirac_delta(k+65) - 311*self.dirac_delta(k+66) - 430*self.dirac_delta(k+67) - 546*self.dirac_delta(k+68) - 659*self.dirac_delta(k+69) - 769*self.dirac_delta(k+70) - 876*self.dirac_delta(k+71) - 980*self.dirac_delta(k+72) - 1081*self.dirac_delta(k+73) - 1179*self.dirac_delta(k+74) - 1274*self.dirac_delta(k+75) - 1366*self.dirac_delta(k+76) - 1455*self.dirac_delta(k+77) - 1541*self.dirac_delta(k+78) - 1624*self.dirac_delta(k+79) - 1704*self.dirac_delta(k+80) - 1781*self.dirac_delta(k+81) - 1855*self.dirac_delta(k+82) - 1926*self.dirac_delta(k+83) - 1994*self.dirac_delta(k+84) - 2059*self.dirac_delta(k+85) - 2121*self.dirac_delta(k+86) - 2180*self.dirac_delta(k+87) - 2236*self.dirac_delta(k+88) - 2289*self.dirac_delta(k+89) - 2339*self.dirac_delta(k+90) - 2386*self.dirac_delta(k+91) - 2430*self.dirac_delta(k+92) - 2471*self.dirac_delta(k+93) - 2509*self.dirac_delta(k+94) - 2544*self.dirac_delta(k+95) - 2576*self.dirac_delta(k+96) - 2605*self.dirac_delta(k+97) - 2631*self.dirac_delta(k+98) - 2654*self.dirac_delta(k+99) - 2674*self.dirac_delta(k+100) - 2691*self.dirac_delta(k+101) - 2705*self.dirac_delta(k+102) - 2716*self.dirac_delta(k+103) - 2724*self.dirac_delta(k+104) - 2729*self.dirac_delta(k+105) - 2731*self.dirac_delta(k+106) - 2730*self.dirac_delta(k+107) - 2726*self.dirac_delta(k+108) - 2719*self.dirac_delta(k+109) - 2709*self.dirac_delta(k+110) - 2696*self.dirac_delta(k+111) - 2680*self.dirac_delta(k+112) - 2661*self.dirac_delta(k+113) - 2639*self.dirac_delta(k+114) - 2614*self.dirac_delta(k+115) - 2586*self.dirac_delta(k+116) - 2555*self.dirac_delta(k+117) - 2521*self.dirac_delta(k+118) - 2484*self.dirac_delta(k+119) - 2444*self.dirac_delta(k+120) - 2401*self.dirac_delta(k+121) - 2355*self.dirac_delta(k+122) - 2306*self.dirac_delta(k+123) - 2254*self.dirac_delta(k+124) - 2199*self.dirac_delta(k+125) - 2141*self.dirac_delta(k+126) - 2080*self.dirac_delta(k+127) - 2016*self.dirac_delta(k+128) - 1953*self.dirac_delta(k+129) - 1891*self.dirac_delta(k+130) - 1830*self.dirac_delta(k+131) - 1770*self.dirac_delta(k+132) - 1711*self.dirac_delta(k+133) - 1653*self.dirac_delta(k+134) - 1596*self.dirac_delta(k+135) - 1540*self.dirac_delta(k+136) - 1485*self.dirac_delta(k+137) - 1431*self.dirac_delta(k+138) - 1378*self.dirac_delta(k+139) - 1326*self.dirac_delta(k+140) - 1275*self.dirac_delta(k+141) - 1225*self.dirac_delta(k+142) - 1176*self.dirac_delta(k+143) - 1128*self.dirac_delta(k+144) - 1081*self.dirac_delta(k+145) - 1035*self.dirac_delta(k+146) - 990*self.dirac_delta(k+147) - 946*self.dirac_delta(k+148) - 903*self.dirac_delta(k+149) - 861*self.dirac_delta(k+150) - 820*self.dirac_delta(k+151) - 780*self.dirac_delta(k+152) - 741*self.dirac_delta(k+153) - 703*self.dirac_delta(k+154) - 666*self.dirac_delta(k+155) - 630*self.dirac_delta(k+156) - 595*self.dirac_delta(k+157) - 561*self.dirac_delta(k+158) - 528*self.dirac_delta(k+159) - 496*self.dirac_delta(k+160) - 465*self.dirac_delta(k+161) - 435*self.dirac_delta(k+162) - 406*self.dirac_delta(k+163) - 378*self.dirac_delta(k+164) - 351*self.dirac_delta(k+165) - 325*self.dirac_delta(k+166) - 300*self.dirac_delta(k+167) - 276*self.dirac_delta(k+168) - 253*self.dirac_delta(k+169) - 231*self.dirac_delta(k+170) - 210*self.dirac_delta(k+171) - 190*self.dirac_delta(k+172) - 171*self.dirac_delta(k+173) - 153*self.dirac_delta(k+174) - 136*self.dirac_delta(k+175) - 120*self.dirac_delta(k+176) - 105*self.dirac_delta(k+177) - 91*self.dirac_delta(k+178) - 78*self.dirac_delta(k+179) - 66*self.dirac_delta(k+180) - 55*self.dirac_delta(k+181) - 45*self.dirac_delta(k+182) - 36*self.dirac_delta(k+183) - 28*self.dirac_delta(k+184) - 21*self.dirac_delta(k+185) - 15*self.dirac_delta(k+186) - 10*self.dirac_delta(k+187) - 6*self.dirac_delta(k+188) - 3*self.dirac_delta(k+189) - self.dirac_delta(k+190))
            elif j == 8:
                for k in range(start_k, end_k): filter_dict[k] = -1/1048576 * (self.dirac_delta(k-127) + 3*self.dirac_delta(k-126) + 6*self.dirac_delta(k-125) + 10*self.dirac_delta(k-124) + 15*self.dirac_delta(k-123) + 21*self.dirac_delta(k-122) + 28*self.dirac_delta(k-121) + 36*self.dirac_delta(k-120) + 45*self.dirac_delta(k-119) + 55*self.dirac_delta(k-118) + 66*self.dirac_delta(k-117) + 78*self.dirac_delta(k-116) + 91*self.dirac_delta(k-115) + 105*self.dirac_delta(k-114) + 120*self.dirac_delta(k-113) + 136*self.dirac_delta(k-112) + 153*self.dirac_delta(k-111) + 171*self.dirac_delta(k-110) + 190*self.dirac_delta(k-109) + 210*self.dirac_delta(k-108) + 231*self.dirac_delta(k-107) + 253*self.dirac_delta(k-106) + 276*self.dirac_delta(k-105) + 300*self.dirac_delta(k-104) + 325*self.dirac_delta(k-103) + 351*self.dirac_delta(k-102) + 378*self.dirac_delta(k-101) + 406*self.dirac_delta(k-100) + 435*self.dirac_delta(k-99) + 465*self.dirac_delta(k-98) + 496*self.dirac_delta(k-97) + 528*self.dirac_delta(k-96) + 561*self.dirac_delta(k-95) + 595*self.dirac_delta(k-94) + 630*self.dirac_delta(k-93) + 666*self.dirac_delta(k-92) + 703*self.dirac_delta(k-91) + 741*self.dirac_delta(k-90) + 780*self.dirac_delta(k-89) + 820*self.dirac_delta(k-88) + 861*self.dirac_delta(k-87) + 903*self.dirac_delta(k-86) + 946*self.dirac_delta(k-85) + 990*self.dirac_delta(k-84) + 1035*self.dirac_delta(k-83) + 1081*self.dirac_delta(k-82) + 1128*self.dirac_delta(k-81) + 1176*self.dirac_delta(k-80) + 1225*self.dirac_delta(k-79) + 1275*self.dirac_delta(k-78) + 1326*self.dirac_delta(k-77) + 1378*self.dirac_delta(k-76) + 1431*self.dirac_delta(k-75) + 1485*self.dirac_delta(k-74) + 1540*self.dirac_delta(k-73) + 1596*self.dirac_delta(k-72) + 1653*self.dirac_delta(k-71) + 1711*self.dirac_delta(k-70) + 1770*self.dirac_delta(k-69) + 1830*self.dirac_delta(k-68) + 1891*self.dirac_delta(k-67) + 1953*self.dirac_delta(k-66) + 2016*self.dirac_delta(k-65) + 2080*self.dirac_delta(k-64) + 2145*self.dirac_delta(k-63) + 2211*self.dirac_delta(k-62) + 2278*self.dirac_delta(k-61) + 2346*self.dirac_delta(k-60) + 2415*self.dirac_delta(k-59) + 2485*self.dirac_delta(k-58) + 2556*self.dirac_delta(k-57) + 2628*self.dirac_delta(k-56) + 2701*self.dirac_delta(k-55) + 2775*self.dirac_delta(k-54) + 2850*self.dirac_delta(k-53) + 2926*self.dirac_delta(k-52) + 3003*self.dirac_delta(k-51) + 3081*self.dirac_delta(k-50) + 3160*self.dirac_delta(k-49) + 3240*self.dirac_delta(k-48) + 3321*self.dirac_delta(k-47) + 3403*self.dirac_delta(k-46) + 3486*self.dirac_delta(k-45) + 3570*self.dirac_delta(k-44) + 3655*self.dirac_delta(k-43) + 3741*self.dirac_delta(k-42) + 3828*self.dirac_delta(k-41) + 3916*self.dirac_delta(k-40) + 4005*self.dirac_delta(k-39) + 4186*self.dirac_delta(k-37) + 4278*self.dirac_delta(k-36) + 4371*self.dirac_delta(k-35) + 4465*self.dirac_delta(k-34) + 4560*self.dirac_delta(k-33) + 4656*self.dirac_delta(k-32) + 4753*self.dirac_delta(k-31) + 4851*self.dirac_delta(k-30) + 4950*self.dirac_delta(k-29) + 5050*self.dirac_delta(k-28) + 5151*self.dirac_delta(k-27) + 5253*self.dirac_delta(k-26) + 5356*self.dirac_delta(k-25) + 5460*self.dirac_delta(k-24) + 5565*self.dirac_delta(k-23) + 5671*self.dirac_delta(k-22) + 5778*self.dirac_delta(k-21) + 5886*self.dirac_delta(k-20) + 5995*self.dirac_delta(k-19) + 6105*self.dirac_delta(k-18) + 6216*self.dirac_delta(k-17) + 6328*self.dirac_delta(k-16) + 6441*self.dirac_delta(k-15) + 6555*self.dirac_delta(k-14) + 6670*self.dirac_delta(k-13) + 6786*self.dirac_delta(k-12) + 6903*self.dirac_delta(k-11) + 7021*self.dirac_delta(k-10) + 7140*self.dirac_delta(k-9) + 7260*self.dirac_delta(k-8) + 7381*self.dirac_delta(k-7) + 7503*self.dirac_delta(k-6) + 7626*self.dirac_delta(k-5) + 7750*self.dirac_delta(k-4) + 7875*self.dirac_delta(k-3) + 8001*self.dirac_delta(k-2) + 8128*self.dirac_delta(k-1) + 8256*self.dirac_delta(k) + 8381*self.dirac_delta(k+1) + 8503*self.dirac_delta(k+2) + 8622*self.dirac_delta(k+3) + 8738*self.dirac_delta(k+4) + 8851*self.dirac_delta(k+5) + 8961*self.dirac_delta(k+6) + 9068*self.dirac_delta(k+7) + 9172*self.dirac_delta(k+8) + 9273*self.dirac_delta(k+9) + 9371*self.dirac_delta(k+10) + 9466*self.dirac_delta(k+11) + 9558*self.dirac_delta(k+12) + 9647*self.dirac_delta(k+13) + 9733*self.dirac_delta(k+14) + 9816*self.dirac_delta(k+15) + 9896*self.dirac_delta(k+16) + 9973*self.dirac_delta(k+17) + 10047*self.dirac_delta(k+18) + 10118*self.dirac_delta(k+19) + 10186*self.dirac_delta(k+20) + 10251*self.dirac_delta(k+21) + 10313*self.dirac_delta(k+22) + 10372*self.dirac_delta(k+23) + 10428*self.dirac_delta(k+24) + 10481*self.dirac_delta(k+25) + 10531*self.dirac_delta(k+26) + 10578*self.dirac_delta(k+27) + 10622*self.dirac_delta(k+28) + 10663*self.dirac_delta(k+29) + 10701*self.dirac_delta(k+30) + 10736*self.dirac_delta(k+31) + 10768*self.dirac_delta(k+32) + 10797*self.dirac_delta(k+33) + 10823*self.dirac_delta(k+34) + 10846*self.dirac_delta(k+35) + 10866*self.dirac_delta(k+36) + 10883*self.dirac_delta(k+37) + 10897*self.dirac_delta(k+38) + 10908*self.dirac_delta(k+39) + 10916*self.dirac_delta(k+40) + 10921*self.dirac_delta(k+41) + 10923*self.dirac_delta(k+42) + 10922*self.dirac_delta(k+43) + 10918*self.dirac_delta(k+44) + 10911*self.dirac_delta(k+45) + 10901*self.dirac_delta(k+46) + 10888*self.dirac_delta(k+47) + 10872*self.dirac_delta(k+48) + 10853*self.dirac_delta(k+49) + 10831*self.dirac_delta(k+50) + 10806*self.dirac_delta(k+51) + 10778*self.dirac_delta(k+52) + 10747*self.dirac_delta(k+53) + 10713*self.dirac_delta(k+54) + 10676*self.dirac_delta(k+55) + 10636*self.dirac_delta(k+56) + 10593*self.dirac_delta(k+57) + 10547*self.dirac_delta(k+58) + 10498*self.dirac_delta(k+59) + 10446*self.dirac_delta(k+60) + 10391*self.dirac_delta(k+61) + 10333*self.dirac_delta(k+62) + 10272*self.dirac_delta(k+63) + 10208*self.dirac_delta(k+64) + 10141*self.dirac_delta(k+65) + 10071*self.dirac_delta(k+66) + 9998*self.dirac_delta(k+67) + 9922*self.dirac_delta(k+68) + 9843*self.dirac_delta(k+69) + 9761*self.dirac_delta(k+70) + 9676*self.dirac_delta(k+71) + 9588*self.dirac_delta(k+72) + 9497*self.dirac_delta(k+73) + 9403*self.dirac_delta(k+74) + 9306*self.dirac_delta(k+75) + 9206*self.dirac_delta(k+76) + 9103*self.dirac_delta(k+77) + 8997*self.dirac_delta(k+78) + 8888*self.dirac_delta(k+79) + 8776*self.dirac_delta(k+80) + 8661*self.dirac_delta(k+81) + 8543*self.dirac_delta(k+82) + 8422*self.dirac_delta(k+83) + 8298*self.dirac_delta(k+84) + 8171*self.dirac_delta(k+85) + 8041*self.dirac_delta(k+86) + 7908*self.dirac_delta(k+87) + 7772*self.dirac_delta(k+88) + 7633*self.dirac_delta(k+89) + 7491*self.dirac_delta(k+90) + 7346*self.dirac_delta(k+91) + 7198*self.dirac_delta(k+92) + 7047*self.dirac_delta(k+93) + 6893*self.dirac_delta(k+94) + 6736*self.dirac_delta(k+95) + 6576*self.dirac_delta(k+96) + 6413*self.dirac_delta(k+97) + 6247*self.dirac_delta(k+98) + 5906*self.dirac_delta(k+100) + 5731*self.dirac_delta(k+101) + 5553*self.dirac_delta(k+102) + 5372*self.dirac_delta(k+103) + 5188*self.dirac_delta(k+104) + 5001*self.dirac_delta(k+105) + 4811*self.dirac_delta(k+106) + 4618*self.dirac_delta(k+107) + 4422*self.dirac_delta(k+108) + 4223*self.dirac_delta(k+109) + 4021*self.dirac_delta(k+110) + 3816*self.dirac_delta(k+111) + 3608*self.dirac_delta(k+112) + 3397*self.dirac_delta(k+113) + 3183*self.dirac_delta(k+114) + 2966*self.dirac_delta(k+115) + 2746*self.dirac_delta(k+116) + 2523*self.dirac_delta(k+117) + 2297*self.dirac_delta(k+118) + 2068*self.dirac_delta(k+119) + 1836*self.dirac_delta(k+120) + 1601*self.dirac_delta(k+121) + 1363*self.dirac_delta(k+122) + 1122*self.dirac_delta(k+123) + 878*self.dirac_delta(k+124) + 631*self.dirac_delta(k+125) + 381*self.dirac_delta(k+126) + 128*self.dirac_delta(k+127) - 128*self.dirac_delta(k+128) - 381*self.dirac_delta(k+129) - 631*self.dirac_delta(k+130) - 878*self.dirac_delta(k+131) - 1122*self.dirac_delta(k+132) - 1363*self.dirac_delta(k+133) - 1601*self.dirac_delta(k+134) - 1836*self.dirac_delta(k+135) - 2068*self.dirac_delta(k+136) - 2297*self.dirac_delta(k+137) - 2523*self.dirac_delta(k+138) - 2746*self.dirac_delta(k+139) - 2966*self.dirac_delta(k+140) - 3183*self.dirac_delta(k+141) - 3397*self.dirac_delta(k+142) - 3608*self.dirac_delta(k+143) - 3816*self.dirac_delta(k+144) - 4021*self.dirac_delta(k+145) - 4223*self.dirac_delta(k+146) - 4422*self.dirac_delta(k+147) - 4618*self.dirac_delta(k+148) - 4811*self.dirac_delta(k+149) - 5001*self.dirac_delta(k+150) - 5188*self.dirac_delta(k+151) - 5372*self.dirac_delta(k+152) - 5553*self.dirac_delta(k+153) - 5731*self.dirac_delta(k+154) - 5906*self.dirac_delta(k+155) - 6078*self.dirac_delta(k+156) - 6247*self.dirac_delta(k+157) - 6413*self.dirac_delta(k+158) - 6576*self.dirac_delta(k+159) - 6736*self.dirac_delta(k+160) - 6893*self.dirac_delta(k+161) - 7047*self.dirac_delta(k+162) - 7198*self.dirac_delta(k+163) - 7346*self.dirac_delta(k+164) - 7491*self.dirac_delta(k+165) - 7633*self.dirac_delta(k+166) - 7772*self.dirac_delta(k+167) - 7908*self.dirac_delta(k+168) - 8041*self.dirac_delta(k+169) - 8171*self.dirac_delta(k+170) - 8298*self.dirac_delta(k+171) - 8422*self.dirac_delta(k+172) - 8543*self.dirac_delta(k+173) - 8661*self.dirac_delta(k+174) - 8776*self.dirac_delta(k+175) - 8888*self.dirac_delta(k+176) - 8997*self.dirac_delta(k+177) - 9103*self.dirac_delta(k+178) - 9206*self.dirac_delta(k+179) - 9306*self.dirac_delta(k+180) - 9403*self.dirac_delta(k+181) - 9497*self.dirac_delta(k+182) - 9588*self.dirac_delta(k+183) - 9676*self.dirac_delta(k+184) - 9761*self.dirac_delta(k+185) - 9843*self.dirac_delta(k+186) - 9922*self.dirac_delta(k+187) - 9998*self.dirac_delta(k+188) - 10071*self.dirac_delta(k+189) - 10141*self.dirac_delta(k+190) - 10208*self.dirac_delta(k+191) - 10272*self.dirac_delta(k+192) - 10333*self.dirac_delta(k+193) - 10391*self.dirac_delta(k+194) - 10446*self.dirac_delta(k+195) - 10498*self.dirac_delta(k+196) - 10547*self.dirac_delta(k+197) - 10593*self.dirac_delta(k+198) - 10636*self.dirac_delta(k+199) - 10676*self.dirac_delta(k+200) - 10713*self.dirac_delta(k+201) - 10747*self.dirac_delta(k+202) - 10778*self.dirac_delta(k+203) - 10806*self.dirac_delta(k+204) - 10831*self.dirac_delta(k+205) - 10853*self.dirac_delta(k+206) - 10872*self.dirac_delta(k+207) - 10888*self.dirac_delta(k+208) - 10901*self.dirac_delta(k+209) - 10911*self.dirac_delta(k+210) - 10918*self.dirac_delta(k+211) - 10922*self.dirac_delta(k+212) - 10923*self.dirac_delta(k+213) - 10921*self.dirac_delta(k+214) - 10916*self.dirac_delta(k+215) - 10908*self.dirac_delta(k+216) - 10897*self.dirac_delta(k+217) - 10883*self.dirac_delta(k+218) - 10866*self.dirac_delta(k+219) - 10846*self.dirac_delta(k+220) - 10823*self.dirac_delta(k+221) - 10797*self.dirac_delta(k+222) - 10768*self.dirac_delta(k+223) - 10736*self.dirac_delta(k+224) - 10701*self.dirac_delta(k+225) - 10663*self.dirac_delta(k+226) - 10622*self.dirac_delta(k+227) - 10578*self.dirac_delta(k+228) - 10531*self.dirac_delta(k+229) - 10481*self.dirac_delta(k+230) - 10428*self.dirac_delta(k+231) - 10372*self.dirac_delta(k+232) - 10313*self.dirac_delta(k+233) - 10251*self.dirac_delta(k+234) - 10186*self.dirac_delta(k+235) - 10118*self.dirac_delta(k+236) - 10047*self.dirac_delta(k+237) - 9973*self.dirac_delta(k+238) - 9896*self.dirac_delta(k+239) - 9816*self.dirac_delta(k+240) - 9733*self.dirac_delta(k+241) - 9647*self.dirac_delta(k+242) - 9558*self.dirac_delta(k+243) - 9466*self.dirac_delta(k+244) - 9371*self.dirac_delta(k+245) - 9273*self.dirac_delta(k+246) - 9172*self.dirac_delta(k+247) - 9068*self.dirac_delta(k+248) - 8961*self.dirac_delta(k+249) - 8851*self.dirac_delta(k+250) - 8738*self.dirac_delta(k+251) - 8622*self.dirac_delta(k+252) - 8503*self.dirac_delta(k+253) - 8381*self.dirac_delta(k+254) - 8256*self.dirac_delta(k+255) - 8128*self.dirac_delta(k+256) - 8001*self.dirac_delta(k+257) - 7875*self.dirac_delta(k+258) - 7750*self.dirac_delta(k+259) - 7626*self.dirac_delta(k+260) - 7503*self.dirac_delta(k+261) - 7381*self.dirac_delta(k+262) - 7260*self.dirac_delta(k+263) - 7140*self.dirac_delta(k+264) - 7021*self.dirac_delta(k+265) - 6903*self.dirac_delta(k+266) - 6786*self.dirac_delta(k+267) - 6670*self.dirac_delta(k+268) - 6555*self.dirac_delta(k+269) - 6441*self.dirac_delta(k+270) - 6328*self.dirac_delta(k+271) - 6216*self.dirac_delta(k+272) - 6105*self.dirac_delta(k+273) - 5995*self.dirac_delta(k+274) - 5886*self.dirac_delta(k+275) - 5778*self.dirac_delta(k+276) - 5671*self.dirac_delta(k+277) - 5565*self.dirac_delta(k+278) - 5460*self.dirac_delta(k+279) - 5356*self.dirac_delta(k+280) - 5253*self.dirac_delta(k+281) - 5151*self.dirac_delta(k+282) - 5050*self.dirac_delta(k+283) - 4950*self.dirac_delta(k+284) - 4851*self.dirac_delta(k+285) - 4753*self.dirac_delta(k+286) - 4656*self.dirac_delta(k+287) - 4560*self.dirac_delta(k+288) - 4465*self.dirac_delta(k+289) - 4371*self.dirac_delta(k+290) - 4278*self.dirac_delta(k+291) - 4186*self.dirac_delta(k+292) - 4095*self.dirac_delta(k+293) - 4005*self.dirac_delta(k+294) - 3916*self.dirac_delta(k+295) - 3828*self.dirac_delta(k+296) - 3741*self.dirac_delta(k+297) - 3655*self.dirac_delta(k+298) - 3570*self.dirac_delta(k+299) - 3486*self.dirac_delta(k+300) - 3403*self.dirac_delta(k+301) - 3321*self.dirac_delta(k+302) - 3240*self.dirac_delta(k+303) - 3160*self.dirac_delta(k+304) - 3081*self.dirac_delta(k+305) - 3003*self.dirac_delta(k+306) - 2926*self.dirac_delta(k+307) - 2850*self.dirac_delta(k+308) - 2775*self.dirac_delta(k+309) - 2701*self.dirac_delta(k+310) - 2628*self.dirac_delta(k+311) - 2556*self.dirac_delta(k+312) - 2485*self.dirac_delta(k+313) - 2415*self.dirac_delta(k+314) - 2346*self.dirac_delta(k+315) - 2278*self.dirac_delta(k+316) - 2211*self.dirac_delta(k+317) - 2145*self.dirac_delta(k+318) - 2080*self.dirac_delta(k+319) - 2016*self.dirac_delta(k+320) - 1953*self.dirac_delta(k+321) - 1891*self.dirac_delta(k+322) - 1830*self.dirac_delta(k+323) - 1770*self.dirac_delta(k+324) - 1711*self.dirac_delta(k+325) - 1653*self.dirac_delta(k+326) - 1596*self.dirac_delta(k+327) - 1540*self.dirac_delta(k+328) - 1485*self.dirac_delta(k+329) - 1431*self.dirac_delta(k+330) - 1378*self.dirac_delta(k+331) - 1326*self.dirac_delta(k+332) - 1275*self.dirac_delta(k+333) - 1225*self.dirac_delta(k+334) - 1176*self.dirac_delta(k+335) - 1128*self.dirac_delta(k+336) - 1081*self.dirac_delta(k+337) - 1035*self.dirac_delta(k+338) - 990*self.dirac_delta(k+339) - 946*self.dirac_delta(k+340) - 903*self.dirac_delta(k+341) - 861*self.dirac_delta(k+342) - 820*self.dirac_delta(k+343) - 780*self.dirac_delta(k+344) - 741*self.dirac_delta(k+345) - 703*self.dirac_delta(k+346) - 666*self.dirac_delta(k+347) - 630*self.dirac_delta(k+348) - 595*self.dirac_delta(k+349) - 561*self.dirac_delta(k+350) - 528*self.dirac_delta(k+351) - 496*self.dirac_delta(k+352) - 465*self.dirac_delta(k+353) - 435*self.dirac_delta(k+354) - 406*self.dirac_delta(k+355) - 378*self.dirac_delta(k+356) - 351*self.dirac_delta(k+357) - 325*self.dirac_delta(k+358) - 300*self.dirac_delta(k+359) - 276*self.dirac_delta(k+360) - 253*self.dirac_delta(k+361) - 231*self.dirac_delta(k+362) - 210*self.dirac_delta(k+363) - 190*self.dirac_delta(k+364) - 171*self.dirac_delta(k+365) - 153*self.dirac_delta(k+366) - 136*self.dirac_delta(k+367) - 120*self.dirac_delta(k+368) - 105*self.dirac_delta(k+369) - 91*self.dirac_delta(k+370) - 78*self.dirac_delta(k+371) - 66*self.dirac_delta(k+372) - 55*self.dirac_delta(k+373) - 45*self.dirac_delta(k+374) - 36*self.dirac_delta(k+375) - 28*self.dirac_delta(k+376) - 21*self.dirac_delta(k+377) - 15*self.dirac_delta(k+378) - 10*self.dirac_delta(k+379) - 6*self.dirac_delta(k+380) - 3*self.dirac_delta(k+381) - self.dirac_delta(k+382))
            qj_filters[j] = filter_dict
        return qj_filters
    
    def calculate_qj_frequency_responses(self, fs):
        h = {n: 1/8 * (self.dirac_delta(n-1) + 3*self.dirac_delta(n) + 3*self.dirac_delta(n+1) + self.dirac_delta(n+2)) for n in range(-2, 2)}
        g = {n: -2 * (self.dirac_delta(n) - self.dirac_delta(n+1)) for n in range(-2, 2)}
        
        # [PLOT FIX] Use a single high-resolution internal FS for generating templates
        internal_fs = 2048 
        num_base_points = int(internal_fs * 128) # Ensure it's large enough for high-order scaling
        Hw, Gw = np.zeros(num_base_points), np.zeros(num_base_points)
        base_freqs = np.linspace(0, internal_fs, num_base_points, endpoint=False)

        for i in range(num_base_points):
            reH, imH, reG, imG = 0, 0, 0, 0
            for k_idx in range(-2, 2):
                angle = 2 * np.pi * base_freqs[i] * k_idx / internal_fs
                reH += h[k_idx] * np.cos(angle); imH -= h[k_idx] * np.sin(angle)
                reG += g[k_idx] * np.cos(angle); imG -= g[k_idx] * np.sin(angle)
            Hw[i] = np.sqrt(reH**2 + imH**2)
            Gw[i] = np.sqrt(reG**2 + imG**2)

        num_final_points = round(fs / 2) + 1
        final_freqs = np.linspace(0, fs / 2, num_final_points)
        Q = np.zeros((9, num_final_points))

        for i in range(num_final_points):
            f = final_freqs[i]
            
            def get_val(arr, freq):
                # Map absolute frequency (Hz) to the correct index in the high-res template
                idx = int(round(freq * num_base_points / internal_fs))
                return arr[idx] if idx < len(arr) else 0

            h_vals = {2**p: get_val(Hw, (2**p)*f) for p in range(7)}
            
            Q[1][i] = get_val(Gw, f)
            Q[2][i] = get_val(Gw, 2*f) * h_vals[1]
            Q[3][i] = get_val(Gw, 4*f) * h_vals[2] * h_vals[1]
            Q[4][i] = get_val(Gw, 8*f) * h_vals[4] * h_vals[2] * h_vals[1]
            Q[5][i] = get_val(Gw, 16*f) * h_vals[8] * h_vals[4] * h_vals[2] * h_vals[1]
            Q[6][i] = get_val(Gw, 32*f) * h_vals[16] * h_vals[8] * h_vals[4] * h_vals[2] * h_vals[1]
            Q[7][i] = get_val(Gw, 64*f) * h_vals[32] * h_vals[16] * h_vals[8] * h_vals[4] * h_vals[2] * h_vals[1]
            Q[8][i] = get_val(Gw, 128*f) * h_vals[64] * h_vals[32] * h_vals[16] * h_vals[8] * h_vals[4] * h_vals[2] * h_vals[1]
        
        return {j: (final_freqs, Q[j]) for j in range(1, 9)}

    def fft_from_scratch(self, signal):
        N = len(signal)
        if N <= 1: return np.array(signal, dtype=np.complex128)
        if N & (N - 1) != 0:
            next_pow2 = 1 << (N - 1).bit_length()
            padded_signal = np.pad(signal, (0, next_pow2 - N), 'constant')
            N = next_pow2
        else:
            padded_signal = np.array(signal, dtype=np.complex128)
        even = self.fft_from_scratch(padded_signal[0::2])
        odd = self.fft_from_scratch(padded_signal[1::2])
        twiddle_factors = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        term = twiddle_factors * odd
        return np.concatenate([even + term, even - term])

    def fft_magnitude_and_frequencies(self, signal):
        fft_complex = self.fft_from_scratch(signal)
        N = len(fft_complex)
        if N == 0: return np.array([]), np.array([])
        magnitude = np.abs(fft_complex)[:N//2] * 2 / N
        if N > 0: magnitude[0] /= 2
        frequencies = np.fft.fftfreq(N, 1.0/self.fs)[:N//2]
        return frequencies, magnitude

    def dwt_convolution_from_scratch(self, signal, max_scale=8):
        dwt_coeffs = {}
        for j in range(1, max_scale + 1):
            if j in self.qj_time_coeffs:
                q_filter_dict = self.qj_time_coeffs[j]
                min_k, max_k = min(q_filter_dict.keys()), max(q_filter_dict.keys())
                filter_coeffs = [q_filter_dict.get(k, 0) for k in range(min_k, max_k + 1)]
                dwt_coeffs[j] = np.convolve(signal, filter_coeffs, mode='same')
        return dwt_coeffs
    
    def load_ppg_data(self, csv_file, ppg_column_name):
        try:
            df = pd.read_csv(csv_file)
            time_col = next((col for col in df.columns if 'time' in col.lower()), None)
            if time_col is None or ppg_column_name not in df.columns:
                 raise ValueError(f"Could not find Time column or '{ppg_column_name}' column.")
            time, ppg_signal = df[time_col].values, df[ppg_column_name].values
            valid_indices = np.isfinite(time) & np.isfinite(ppg_signal)
            time, ppg_signal = time[valid_indices], ppg_signal[valid_indices]
            if len(time) < 2: raise ValueError("Not enough valid data points.")
            self.original_fs = self.fs = len(time) / (time[-1] - time[0])
            return time, ppg_signal
        except Exception as e:
            print(f"Error loading data: {e}"); return None, None

    def downsample_signal(self, signal, time, factor):
        if factor <= 1:
            self.fs = self.original_fs
            return signal, time
        self.fs = self.original_fs / factor
        return signal[::factor], time[::factor]

class HRV_Analyzer:
    def __init__(self, signal, time_vector, fs, fft_function):
        self.raw_signal, self.time, self.fs, self.fft_func = np.array(signal), np.array(time_vector), fs, fft_function
        self.preprocessed_signal, self.peaks, self.minima, self.rr_intervals = None, None, None, None

    def _preprocess_and_filter(self, lowcut=0.5, highcut=4.0):
        window_size = int(1.0 * self.fs)
        if window_size % 2 == 0: window_size += 1
        signal_no_baseline = self.raw_signal - np.convolve(self.raw_signal, np.ones(window_size)/window_size, mode='same') if len(self.raw_signal) > window_size else self.raw_signal
        signal_no_dc = signal_no_baseline - np.mean(signal_no_baseline)
        
        filtered_signal = signal_no_dc
        if (0.5 * self.fs) > highcut:
            b, a = manual_butter_bandpass(lowcut, highcut, self.fs, order=2)
            filtered_signal = manual_lfilter(b, a, signal_no_dc)
        
        # [AMPLITUDE FIX] Normalize the output signal to have a std of 1
        sig_std = np.std(filtered_signal)
        if sig_std > 0:
            self.preprocessed_signal = filtered_signal / sig_std
        else:
            self.preprocessed_signal = filtered_signal

    def _detect_peaks(self):
        if self.preprocessed_signal is None: return
        # Thresholds are now robust due to normalization in preprocessing
        peaks = manual_find_peaks(self.preprocessed_signal, distance=int(self.fs * 0.33), height=0.3, prominence=0.3)
        if len(peaks) < 2: self.peaks = peaks; return
        rr_s = np.diff(self.time[peaks])
        median_rr = np.median(rr_s)
        final_mask = np.ones_like(peaks, dtype=bool)
        for i in range(len(rr_s)):
            if not (0.5 * median_rr < rr_s[i] < 1.5 * median_rr):
                final_mask[i if self.preprocessed_signal[peaks[i]] < self.preprocessed_signal[peaks[i+1]] else i+1] = False
        self.peaks = peaks[final_mask]
        if len(self.peaks) > 1: self.rr_intervals = np.diff(self.time[self.peaks])

    def _detect_minima(self):
        if self.peaks is None or len(self.peaks) < 2: self.minima = np.array([], dtype=int); return
        minima_indices = [self.peaks[i] + np.argmin(self.preprocessed_signal[self.peaks[i]:self.peaks[i+1]]) for i in range(len(self.peaks) - 1) if self.peaks[i+1] > self.peaks[i]]
        self.minima = np.array(minima_indices, dtype=int)

    def _calculate_time_domain_features(self):
        results = defaultdict(lambda: 0); results['rr_histogram'] = (np.array([]), np.array([]))
        if self.rr_intervals is None or len(self.rr_intervals) < 2: return results
        rr_ms = self.rr_intervals * 1000
        mean_nn = np.mean(rr_ms)
        nn50 = np.sum(np.abs(np.diff(rr_ms)) > 50)
        results.update({'mean_nn': mean_nn, 'sdnn': np.std(rr_ms, ddof=1), 'sdsd': np.std(np.diff(rr_ms), ddof=1), 'rmssd': np.sqrt(np.mean(np.diff(rr_ms)**2)), 'nn50': nn50, 'pnn50': (nn50 / (len(rr_ms) - 1)) * 100 if len(rr_ms) > 1 else 0, 'mean_hr': 60000 / mean_nn if mean_nn > 0 else 0})
        if mean_nn > 0: results['cvnn'], results['cvsd'] = results['sdnn'] / mean_nn, results['rmssd'] / mean_nn
        if results['sdnn'] > 0: results['skewness'] = np.sum(((rr_ms - mean_nn) / results['sdnn']) ** 3) / len(rr_ms)
        if len(rr_ms) > 1:
            hist_c, hist_b = np.histogram(rr_ms, bins=np.arange(np.min(rr_ms), np.max(rr_ms) + 7.8125, 7.8125)); results['rr_histogram'] = (hist_c, hist_b)
            if hist_c.size > 0 and (h_max := np.max(hist_c)) > 0: results['hti'] = len(rr_ms) / h_max
            if (nz_idx := np.where(hist_c > 0)[0]).size > 1: results['tinn'] = hist_b[nz_idx[-1] + 1] - hist_b[nz_idx[0]]
        peak_times = self.time[self.peaks]
        if len(peak_times) > 1 and (peak_times[-1] - peak_times[0]) > 300:
            s_means, s_stds, s_start = [], [], peak_times[0]
            while s_start < peak_times[-1]:
                s_end = s_start + 300
                rr_idx = [i for i, pk_idx in enumerate(self.peaks[1:]) if s_start <= self.time[self.peaks[i+1]] < s_end]
                if len(rr_idx) > 1: s_means.append(np.mean(self.rr_intervals[rr_idx]*1000)); s_stds.append(np.std(self.rr_intervals[rr_idx]*1000, ddof=1))
                s_start = s_end
            if len(s_means) > 1: results['sdann'] = np.std(s_means, ddof=1)
            if len(s_stds) > 0: results['sdnn_index'] = np.mean(s_stds)
        return results

    def _calculate_frequency_domain_features(self, interp_fs=4.0):
        defaults = defaultdict(lambda: 0, {'psd_freqs': None, 'psd_values': None})
        if self.rr_intervals is None or len(self.rr_intervals) < 8: return defaults
        peak_times = self.time[self.peaks][1:]
        interp_time = np.arange(peak_times[0], peak_times[-1], 1.0 / interp_fs)
        if len(interp_time) < 2: return defaults
        interp_rr = np.interp(interp_time, peak_times, self.rr_intervals)
        freqs, psd = welch_from_scratch(interp_rr - np.mean(interp_rr), interp_fs, segment_len=min(256, len(interp_rr)), fft_func=self.fft_func)
        if freqs.size == 0: return defaults
        lf_mask, hf_mask = (freqs >= 0.04) & (freqs < 0.15), (freqs >= 0.15) & (freqs < 0.4)
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        total_power = np.trapz(psd[freqs < 0.4], freqs[freqs < 0.4]) if np.any(freqs < 0.4) else 0
        lf_hf_sum = lf_power + hf_power
        return {'psd_freqs': freqs, 'psd_values': psd, 'total_power': total_power, 'lf_power': lf_power, 'hf_power': hf_power, 'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else np.inf, 'lf_nu': (lf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else 0, 'hf_nu': (hf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else 0, 'peak_lf': freqs[lf_mask][np.argmax(psd[lf_mask])] if np.any(lf_mask) and np.any(psd[lf_mask]) else 0, 'peak_hf': freqs[hf_mask][np.argmax(psd[hf_mask])] if np.any(hf_mask) and np.any(psd[hf_mask]) else 0}

    def _calculate_nonlinear_features(self):
        if self.rr_intervals is None or len(self.rr_intervals) < 2: return {'poincare_x':[],'poincare_y':[],'sd1':0,'sd2':0,'sd1_sd2_ratio':0}
        rr_n, rr_n1 = self.rr_intervals[:-1], self.rr_intervals[1:]
        sd1, sd2 = np.std(np.subtract(rr_n, rr_n1)/np.sqrt(2), ddof=1), np.std(np.add(rr_n, rr_n1)/np.sqrt(2), ddof=1)
        return {'poincare_x': rr_n * 1000, 'poincare_y': rr_n1 * 1000, 'sd1': sd1 * 1000, 'sd2': sd2 * 1000, 'sd1_sd2_ratio': sd1 / sd2 if sd2 > 0 else 0}

    def run_all_analyses(self):
        self._preprocess_and_filter()
        self._detect_peaks()
        self._detect_minima()
        return {"peaks": self.peaks, "minima": self.minima, "rr_intervals_s": self.rr_intervals, "rr_times": self.time[self.peaks] if self.peaks is not None and len(self.peaks) > 0 else np.array([]), "time_domain": self._calculate_time_domain_features(), "frequency_domain": self._calculate_frequency_domain_features(), "nonlinear": self._calculate_nonlinear_features()}