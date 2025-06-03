import os

import librosa
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

class CrossCorrAngleEstimator(nn.Module):
    def __init__(self, max_delay=20, fs=48000, d=0.1, c=343.0):
        super().__init__()
        self.max_delay = max_delay
        self.fs = fs
        self.d = d
        self.c = c
        self.register_buffer('lags', torch.arange(-max_delay, max_delay + 1, dtype=torch.int64))

    def forward(self, audio):
        # audio: (batch, 2, window_size)
        batch, chans, win = audio.shape
        assert chans == 2
        corr = []
        for lag in self.lags:
            if lag >= 0:
                c = (audio[:, 0, lag:] * audio[:, 1, :-lag or None]).sum(dim=1)
            else:
                c = (audio[:, 0, :lag] * audio[:, 1, -lag:]).sum(dim=1)
            corr.append(c)
        corr = torch.stack(corr, dim=1)  # (batch, num_lags)
        max_idx = corr.abs().argmax(dim=1)
        optimal_lag = self.lags[max_idx]
        # Convert lag to angle
        sin_theta = (optimal_lag.float() * self.c) / (self.d * self.fs)
        sin_theta = torch.clamp(sin_theta, -1, 1)
        theta = 90 - torch.arcsin(sin_theta) * 180 / np.pi  # degrees, 0 = you, 180 = opposite you
        # gain is zero if you, 1 if opposite you
        # Confidence (peak-to-second-peak ratio)
        corr_abs = corr.abs()
        sorted_vals, _ = torch.sort(corr_abs, descending=True)
        confidence = sorted_vals[:, 0] / (sorted_vals[:, 1] + 1e-9)
        confidence = torch.tanh(confidence/2)  # squashes to [0,1)

        return optimal_lag, theta, confidence

# --- Streaming / Sliding Window Demo Function ---
def stream_aoa_estimation(audio, fs, d, window_size=512, hop_size=128, max_delay=20):
    """
    audio: stereo np.ndarray (shape [N, 2])
    Returns: times, angles, confidences (np.ndarray)
    """
    estimator = CrossCorrAngleEstimator(max_delay=max_delay, fs=fs, d=d)
    num_samples = audio.shape[0]
    windows = []
    times = []
    idx = 0
    while idx + window_size <= num_samples:
        chunk = audio[idx:idx+window_size, :].T[None, :, :]  # (1, 2, window)
        chunk_torch = torch.from_numpy(chunk.astype(np.float32))
        lag, angle, conf = estimator(chunk_torch)
        windows.append((lag.item(), angle.item(), conf.item()))
        times.append((idx + window_size//2)/fs)
        idx += hop_size
    windows = np.array(windows)
    times = np.array(times)
    delays = windows[:,0]
    angles = windows[:,1]
    confidences = windows[:,2]
    return times, delays, angles, confidences

def expand_gains_to_audio_length(gains, audio_len, hop_size):
    # Time centers of each gain
    gain_times = np.arange(len(gains)) * hop_size + hop_size // 2
    sample_times = np.arange(audio_len)
    # Ensure gains cover full audio range
    if gain_times[0] > 0:
        gain_times = np.insert(gain_times, 0, 0)
        gains = np.insert(gains, 0, gains[0])
    if gain_times[-1] < audio_len - 1:
        gain_times = np.append(gain_times, audio_len - 1)
        gains = np.append(gains, gains[-1])
    # Interpolate
    gain_per_sample = np.interp(sample_times, gain_times, gains)
    return gain_per_sample

# --- Example: Run on a stereo wav file ---
if __name__ == "__main__":
    fs = 16000 # Sampling frequency
    # Load stereo audio (replace 'stereo_example.wav' with your file)
    audio_file = "recordingInput_030625_164504.wav"
    audio_file = "both_in.wav"
    audio, fs_loaded = librosa.load(audio_file, sr=fs, mono=False)
    audio = audio.T  # Transpose to shape (samples, channels)
    assert audio.shape[1] == 2
    # flip channels

    audio = audio[:, [1, 0]]

    # Set physical parameters
    d = 0.204  # meters (20.4 cm mic spacing, adjust as needed)
    # Parameters
    window_size = 8000
    hop_size = 960
    max_delay = 20  # for +/-20 samples

    # Run streaming estimator
    times, delays, angles, confidences = stream_aoa_estimation(
        audio, fs, d, window_size=window_size, hop_size=hop_size, max_delay=max_delay
    )

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(3,1,1)
    plt.plot(np.arange(audio.shape[0])/fs, audio[:,1], label="Front (left)")
    plt.plot(np.arange(audio.shape[0])/fs, audio[:,0], label="Back (right)", alpha=0.6)
    plt.legend()
    plt.title("Stereo Input Waveform")
    plt.subplot(3,1,2)
    plt.plot(times, angles, '.-')
    plt.ylabel("Angle of Arrival (deg)")
    plt.title("Estimated AoA over Time")
    plt.ylim(-0, 180)
    plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(times, confidences, '.-')
    plt.ylabel("Confidence")
    plt.xlabel("Time (s)")
    plt.title("Peak Strength (Normalized Confidence)")
    plt.tight_layout()
    plt.show()

    gains = angles / 180

    gain_per_sample = expand_gains_to_audio_length(gains, len(audio), hop_size)
    audio_out = np.mean(audio, axis=1) * gain_per_sample

    base, ext = os.path.splitext(audio_file)
    output_file = f"{base}_out{ext}"

    sf.write(output_file, audio_out, fs)
    print(f"Saved output audio to: {output_file}")
