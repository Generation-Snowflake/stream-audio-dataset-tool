"""
Audio analysis utilities.
Provides mel spectrogram generation, energy calculation, and temporal variance.
Adapted from sound-analysis project.
"""

import numpy as np
import librosa
import librosa.display

# Default analysis parameters
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


def load_audio_for_analysis(path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """
    Load audio file and normalize to target duration for analysis.

    Args:
        path: Path to audio file
        sr: Sample rate (default: 16000)
        duration: Target duration in seconds (default: 3.0)

    Returns:
        Audio signal as numpy array
    """
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y


def mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE,
                    n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                    n_mels: int = N_MELS) -> np.ndarray:
    """
    Generate mel spectrogram from audio signal.

    Args:
        y: Audio signal as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel filter banks

    Returns:
        Mel spectrogram in dB scale
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)


def avg_energy(mel: np.ndarray) -> float:
    """
    Calculate average energy from mel spectrogram.

    Args:
        mel: Mel spectrogram array (dB scale)

    Returns:
        Average energy in dB
    """
    return float(np.mean(mel))


def temporal_variance(mel: np.ndarray) -> float:
    """
    Calculate temporal variance from mel spectrogram.
    Measures how much the spectral content changes over time.

    Args:
        mel: Mel spectrogram array (dB scale)

    Returns:
        Mean temporal variance
    """
    return float(np.mean(np.var(mel, axis=1)))


def analyze_audio_file(file_path: str, sr: int = SAMPLE_RATE,
                       duration: float = DURATION) -> dict:
    """
    Perform complete analysis on an audio file.

    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Target duration

    Returns:
        Dictionary with analysis results:
        - audio: raw audio signal
        - mel: mel spectrogram
        - energy: average energy (dB)
        - variance: temporal variance
        - duration: actual duration in seconds
        - rms: RMS value
    """
    y = load_audio_for_analysis(file_path, sr=sr, duration=duration)
    mel = mel_spectrogram(y, sr=sr)
    energy = avg_energy(mel)
    variance = temporal_variance(mel)
    rms = float(np.sqrt(np.mean(y ** 2)))

    return {
        'audio': y,
        'mel': mel,
        'energy': energy,
        'variance': variance,
        'duration': len(y) / sr,
        'rms': rms,
        'max_amplitude': float(np.max(np.abs(y))),
        'sample_rate': sr,
    }


def classify_by_threshold(energy: float, threshold: float = -30.0) -> str:
    """
    Classify audio as OK or NG based on energy threshold.

    Args:
        energy: Average energy in dB
        threshold: Energy threshold in dB (default: -30.0)

    Returns:
        'OK' if energy below threshold, 'NG' if above
    """
    if energy >= threshold:
        return 'NG'
    else:
        return 'OK'
