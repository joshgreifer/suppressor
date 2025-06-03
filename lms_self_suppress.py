import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys

from scipy.signal import stft, istft


def simple_dereverb(audio, fs, frame_len=1024, frame_hop=256, alpha=1):
    """
    Simple dereverberation via spectral subtraction of a running minimum.
    alpha: How much of late energy to subtract (0=none, 1=aggressive).
    """
    # STFT
    f, t, Zxx = stft(audio, fs=fs, nperseg=frame_len, noverlap=frame_len-frame_hop)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    # Estimate late energy floor (minimum over trailing window)
    min_win = 10  # number of frames for min-energy floor (adjust as needed)
    mag_min = np.minimum.accumulate(mag, axis=1)
    mag_sub = np.maximum(mag - alpha * mag_min, 0)
    # Recombine and iSTFT
    Zxx_clean = mag_sub * np.exp(1j * phase)
    _, audio_clean = istft(Zxx_clean, fs=fs, nperseg=frame_len, noverlap=frame_len-frame_hop)
    # Truncate to original length
    return audio_clean[:len(audio)]

def fractional_delay(sig, delay):
    """Fractional sample delay using linear interpolation."""
    idx = np.arange(len(sig))
    idx_delayed = idx - delay
    idx0 = np.floor(idx_delayed).astype(int)
    idx1 = idx0 + 1
    w1 = idx_delayed - idx0
    valid = (idx0 >= 0) & (idx1 < len(sig))
    y = np.zeros_like(sig)
    y[valid] = (1 - w1[valid]) * sig[idx0[valid]] + w1[valid] * sig[idx1[valid]]
    return y

def lms_filter(x, d, M=32, mu=0.005):
    """Run LMS adaptive filter: x is reference, d is desired. Returns output (d - y)."""
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for n in range(M, N):
        x_vec = x[n-M:n][::-1]
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w = w + 2 * mu * e[n] * x_vec
    return e

def process_file(input_wav, delay, output_wav, M=32, mu=0.005, plot=True):
    # Load stereo file
    audio, fs = sf.read(input_wav)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Input must be stereo (2 channels).")
    front = audio[:, 0]
    back = audio[:, 1]

    front = simple_dereverb(front, fs)  # Optional dereverberation
    back = simple_dereverb(back, fs)    # Optional dereverberation

    # Apply fractional delay to back channel
    back_delayed = fractional_delay(back, delay)
    # LMS filtering: suppress self (front) from delayed back
    out = lms_filter(front, back_delayed, M=M, mu=mu)
    # Write output
    sf.write(output_wav, out, fs)
    print(f"Wrote suppressed output to {output_wav}")

    # Compute "likelihood" of self speech: Short-time energy of front in windows
    win_len = int(0.2 * fs)      # 200 ms
    hop = int(0.05 * fs)         # 50 ms
    frames = []
    likelihood = []
    waveform = []
    times = []
    for i in range(0, len(out) - win_len, hop):
        seg_front = front[i:i+win_len]
        seg_out = out[i:i+win_len]
        energy_self = np.mean(seg_front**2)
        energy_resid = np.mean(seg_out**2)
        # Likelihood: ratio of self energy remaining in output (inverted so high means more self present)
        L = energy_self / (energy_resid + 1e-8)
        likelihood.append(L)
        waveform.append(np.mean(seg_out))
        times.append((i + win_len // 2) / fs)
    likelihood = np.array(likelihood)
    times = np.array(times)

    # Plot
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(2,1,1)
        plt.plot(np.arange(len(out))/fs, out, label="Suppressed waveform")
        plt.xlim(0, len(out)/fs)
        plt.title("Suppressed Output Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.subplot(2,1,2)
        plt.plot(times, likelihood, label="Self speech likelihood (higher=more self)")
        plt.title("Self-Speaker 'Likelihood' Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Likelihood")
        plt.tight_layout()
        plt.show()
    return out

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python lms_self_suppress.py input.wav delay_in_samples output.wav")
        sys.exit(1)
    input_wav = sys.argv[1]
    delay = float(sys.argv[2])
    output_wav = sys.argv[3]
    process_file(input_wav, delay, output_wav)
