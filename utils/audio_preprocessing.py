"""
Audio preprocessing utilities.
Handles loading, resampling, and normalizing audio files to 16kHz mono format.
Adapted from yam-sound-dectection-pipeline.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Optional


def load_and_preprocess_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and preprocess to 16kHz mono format.

    Args:
        file_path: Path to audio file (.wav, .mp3, etc.)
        target_sr: Target sample rate (default: 16000 Hz for YAMNet)

    Returns:
        Preprocessed audio as numpy array (float32, range: -1 to 1)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Load audio file with librosa (handles multiple formats)
        audio, sr = librosa.load(file_path, sr=None, mono=False)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Resample to target sample rate if needed
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)

        # Normalize audio
        audio = normalize_audio(audio)

        return audio.astype(np.float32)

    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio signal as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio signal
    """
    if orig_sr == target_sr:
        return audio

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio amplitude.

    Args:
        audio: Audio signal as numpy array
        method: Normalization method ('peak' or 'rms')

    Returns:
        Normalized audio signal (range: -1 to 1)
    """
    if len(audio) == 0:
        return audio

    if method == 'peak':
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio / (rms * 10)
            audio = np.clip(audio, -1.0, 1.0)

    return audio


def validate_audio_format(file_path: str) -> bool:
    """
    Check if file is a valid audio file.

    Args:
        file_path: Path to audio file

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False

    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in valid_extensions:
        return False

    try:
        sf.info(file_path)
        return True
    except Exception:
        return False


def pad_or_trim_audio(audio: np.ndarray, target_length: int, sr: int = 16000) -> np.ndarray:
    """
    Pad or trim audio to target length.

    Args:
        audio: Audio signal
        target_length: Target length in seconds
        sr: Sample rate

    Returns:
        Audio padded or trimmed to target length
    """
    target_samples = target_length * sr

    if len(audio) < target_samples:
        padding = target_samples - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain to audio signal.

    Args:
        audio: Audio signal
        gain_db: Gain in decibels

    Returns:
        Audio with applied gain
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear
