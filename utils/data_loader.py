"""
Data loading utilities for OK/NG sound classification dataset.
Adapted from yam-sound-dectection-pipeline.
"""

import os
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(data_dir: str,
                 val_split: float = 0.2,
                 random_state: int = 42) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load dataset from directory structure.

    Expected structure:
        data_dir/
            OK/
                *.wav
            NG/
                *.wav

    Args:
        data_dir: Root directory containing OK and NG subdirectories
        val_split: Validation split ratio (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_files, train_labels, val_files, val_labels)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    classes = ['OK', 'NG']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    file_paths = []
    labels = []

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        if len(files) == 0:
            logger.warning(f"No .wav files found in {class_dir}")
            continue

        logger.info(f"Found {len(files)} files for class '{class_name}'")

        for file_name in files:
            file_path = os.path.join(class_dir, file_name)
            file_paths.append(file_path)
            labels.append(class_to_idx[class_name])

    if len(file_paths) == 0:
        raise ValueError(f"No audio files found in {data_dir}")

    logger.info(f"Total files loaded: {len(file_paths)}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if val_split > 0:
        train_files, val_files, train_labels, val_labels = train_test_split(
            file_paths, labels,
            test_size=val_split,
            random_state=random_state,
            stratify=labels
        )

        logger.info(f"Train set: {len(train_files)} files")
        logger.info(f"Validation set: {len(val_files)} files")

        return train_files, train_labels, val_files, val_labels
    else:
        return file_paths, labels, [], []


def get_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: List of integer labels

    Returns:
        Dictionary mapping class index to weight
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    weights = {int(cls): total / (len(unique) * count)
               for cls, count in zip(unique, counts)}

    logger.info(f"Class weights: {weights}")
    return weights


def create_embedding_dataset(file_paths: List[str],
                             labels: List[int],
                             feature_extractor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings for all files and create dataset.

    Args:
        file_paths: List of audio file paths
        labels: List of corresponding labels
        feature_extractor: YAMNetFeatureExtractor instance

    Returns:
        Tuple of (embeddings, labels) as numpy arrays
    """
    from utils.audio_preprocessing import load_and_preprocess_audio

    embeddings_list = []
    valid_labels = []

    logger.info(f"Extracting embeddings for {len(file_paths)} files...")

    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        try:
            audio = load_and_preprocess_audio(file_path)
            embedding = feature_extractor.extract_mean_embedding(audio)

            embeddings_list.append(embedding)
            valid_labels.append(label)

            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(file_paths)} files")

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {str(e)}")
            continue

    embeddings = np.array(embeddings_list)
    labels_array = np.array(valid_labels)

    logger.info(f"✓ Extracted {len(embeddings)} embeddings")
    logger.info(f"Embedding shape: {embeddings.shape}")

    return embeddings, labels_array


def augment_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Apply random data augmentation to audio.

    Args:
        audio: Audio waveform
        sr: Sample rate

    Returns:
        Augmented audio
    """
    import librosa

    aug_type = np.random.choice(['time_stretch', 'pitch_shift', 'noise', 'none'])

    if aug_type == 'time_stretch':
        rate = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    elif aug_type == 'pitch_shift':
        n_steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    elif aug_type == 'noise':
        noise_level = np.random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_level, len(audio))
        audio = audio + noise

    return audio
