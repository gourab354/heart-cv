"""
rppg_lib.py - Shared Signal-Processing Functions (ECG + rPPG).
Prepared for Gourab - Build Club AOT: Webcam Heart Rate (rPPG) vs ECG.

Implements:
  - 3 rPPG extraction methods: Green channel, CHROM, POS
  - Zero-phase Butterworth filtering
  - Hann-windowed zero-padded FFT frequency estimation
  - Pan-Tompkins QRS R-peak detection on AD8232 ECG
  - Windowed metric extraction for statistical comparison
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

BAND = (0.7, 4.0)  # Hz -> 42..240 BPM


# ----------------------------------------------------------------- Basics
def bandpass(x, fs, low, high, order=3, axis=0):
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_norm = max(1e-4, min(low / nyq, 0.98))
    high_norm = max(low_norm + 1e-3, min(high / nyq, 0.99))
    b, a = signal.butter(order, [low_norm, high_norm], btype="band")
    return signal.filtfilt(b, a, x, axis=axis)


def resample_uniform(t, x, fs):
    """Linear interpolation of (possibly jittery) samples onto a uniform grid."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    keep = np.concatenate(([True], np.diff(t) > 0))  # strictly increasing
    t, x = t[keep], x[keep]
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    return tu, np.column_stack([np.interp(tu, t, x[:, i]) for i in range(x.shape[1])])


# ----------------------------------------------------------------- rPPG Methods
def rppg_green(rgb, fs):
    """Baseline: green channel, normalised and band-passed."""
    g = rgb[:, 1]
    return bandpass(g / g.mean() - 1.0, fs, *BAND)


def rppg_chrom(rgb, fs, win_s=1.6):
    """
    CHROM (de Haan & Jeanne 2013). Chrominance signals are band-passed,
    alpha is computed per 1.6 s window, windows are Hann overlap-added.
    """
    n = len(rgb)
    L = int(round(win_s * fs))
    ma = uniform_filter1d(rgb, size=L, axis=0, mode="nearest")
    cn = rgb / (ma + 1e-12)  # temporal normalisation
    xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf = bandpass(xs, fs, *BAND)
    yf = bandpass(ys, fs, *BAND)
    out = np.zeros(n)
    w = signal.windows.hann(L, sym=False)
    step = L // 2
    for s in range(0, n - L + 1, step):
        xw, yw = xf[s : s + L], yf[s : s + L]
        alpha = xw.std() / (yw.std() + 1e-12)
        sw = xw - alpha * yw
        out[s : s + L] += w * (sw - sw.mean())
    return out


def rppg_pos(rgb, fs, win_s=1.6):
    """
    POS (Wang et al. 2017): plane-orthogonal-to-skin projection with
    overlap-add of 1.6 s windows.
    """
    n = len(rgb)
    L = int(round(win_s * fs))
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h_all = np.zeros(n)
    for s in range(0, n - L + 1):
        c = rgb[s : s + L].T  # 3 x L
        cn = c / (c.mean(axis=1, keepdims=True) + 1e-12)
        sp = P @ cn
        h = sp[0] + (sp[0].std() / (sp[1].std() + 1e-12)) * sp[1]
        h_all[s : s + L] += h - h.mean()
    return bandpass(h_all, fs, *BAND)


METHODS = {"Green": rppg_green, "CHROM": rppg_chrom, "POS": rppg_pos}


def bpm_from_signal(x, fs, band=BAND, nfft=2**15):
    """Dominant spectral peak inside the heart-rate band, in BPM."""
    x = np.asarray(x, float)
    if len(x) < 4 * fs or not np.isfinite(x).all():
        return np.nan
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (f >= band[0]) & (f <= band[1])
    return 60.0 * f[m][np.argmax(spec[m])]


def windowed_bpm(t, x, fs, starts, win_s):
    """Extract BPM in sliding windows."""
    out = []
    for s in starts:
        m = (t >= s) & (t < s + win_s)
        out.append(bpm_from_signal(x[m], fs))
    return np.array(out)


# ----------------------------------------------------------------- ECG Processing
def ecg_r_peaks(t, adc, fs=250):
    """
    Pan-Tompkins-style R-peak detector.
    Returns (peak_times, t_uniform, ecg_uniform).
    """
    tu, x = resample_uniform(t, adc, fs)
    x = x - np.median(x)
    xb = bandpass(x, fs, 5.0, 15.0, order=2)
    d = np.gradient(xb)
    w = max(1, int(0.12 * fs))
    mwi = np.convolve(d**2, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(mwi, 99)
    pk, _ = signal.find_peaks(mwi, height=thr, distance=int(0.3 * fs))
    half = int(0.08 * fs)
    refined = []
    for p in pk:
        a, b = max(0, p - half), min(len(xb), p + half)
        refined.append(a + int(np.argmax(np.abs(xb[a:b]))))
    refined = np.unique(refined)
    return tu[refined], tu, x


def ecg_bpm_windows(peak_t, starts, win_s=30.0):
    """Mean heart rate (BPM) in each window from ECG RR intervals."""
    out = []
    for s in starts:
        p = peak_t[(peak_t >= s) & (peak_t < s + win_s)]
        if len(p) < 8:
            out.append(np.nan)
            continue
        rr = np.diff(p)
        rr = rr[(rr > 0.3) & (rr < 1.5)]  # drop missed / double beats
        out.append(60.0 / rr.mean() if len(rr) >= 6 else np.nan)
    return np.array(out)


def leadoff_fraction(t, lo, starts, win_s=30.0):
    """Calculates lead-off electrode disconnect percentage in window."""
    out = []
    for s in starts:
        m = (t >= s) & (t < s + win_s)
        out.append(lo[m].mean() if m.any() else 1.0)
    return np.array(out)
