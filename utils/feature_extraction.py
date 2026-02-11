"""
YAMNet feature extraction utilities.
Loads pre-trained YAMNet model and extracts embeddings for audio classification.
Adapted from yam-sound-dectection-pipeline.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YAMNetFeatureExtractor:
    """
    Wrapper class for YAMNet feature extraction.
    Loads pre-trained YAMNet model from TensorFlow Hub and extracts 1024-dim embeddings.
    """

    def __init__(self, model_url: str = 'https://tfhub.dev/google/yamnet/1'):
        """
        Initialize YAMNet feature extractor.

        Args:
            model_url: TensorFlow Hub URL for YAMNet model
        """
        logger.info("Loading YAMNet model from TensorFlow Hub...")
        try:
            self.model = hub.load(model_url)
            logger.info("✓ YAMNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YAMNet model: {str(e)}")
            raise

    def extract_embeddings(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from audio waveform.

        Args:
            audio_waveform: Preprocessed audio (16kHz mono, float32)

        Returns:
            Embeddings array of shape (num_frames, 1024)
        """
        waveform = tf.convert_to_tensor(audio_waveform, dtype=tf.float32)
        _, embeddings, _ = self.model(waveform)
        return embeddings.numpy()

    def extract_mean_embedding(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Extract mean embedding across all frames.

        Args:
            audio_waveform: Preprocessed audio (16kHz mono, float32)

        Returns:
            Mean embedding of shape (1024,)
        """
        embeddings = self.extract_embeddings(audio_waveform)
        return np.mean(embeddings, axis=0)

    def extract_max_embedding(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Extract max-pooled embedding across all frames.

        Args:
            audio_waveform: Preprocessed audio (16kHz mono, float32)

        Returns:
            Max-pooled embedding of shape (1024,)
        """
        embeddings = self.extract_embeddings(audio_waveform)
        return np.max(embeddings, axis=0)

    def extract_embeddings_batch(self, audio_waveforms: List[np.ndarray],
                                  pooling: str = 'mean') -> np.ndarray:
        """
        Extract embeddings for a batch of audio waveforms.

        Args:
            audio_waveforms: List of preprocessed audio waveforms
            pooling: Pooling method ('mean' or 'max')

        Returns:
            Batch of embeddings, shape (batch_size, 1024)
        """
        embeddings_list = []

        for audio in audio_waveforms:
            if pooling == 'mean':
                embedding = self.extract_mean_embedding(audio)
            elif pooling == 'max':
                embedding = self.extract_max_embedding(audio)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")

            embeddings_list.append(embedding)

        return np.array(embeddings_list)


def extract_embeddings_from_file(file_path: str,
                                  extractor: YAMNetFeatureExtractor = None,
                                  pooling: str = 'mean') -> np.ndarray:
    """
    Helper function to extract embeddings directly from audio file.

    Args:
        file_path: Path to audio file
        extractor: YAMNetFeatureExtractor instance (creates new if None)
        pooling: Pooling method ('mean' or 'max')

    Returns:
        Audio embeddings, shape (1024,)
    """
    from utils.audio_preprocessing import load_and_preprocess_audio

    audio = load_and_preprocess_audio(file_path)

    if extractor is None:
        extractor = YAMNetFeatureExtractor()

    if pooling == 'mean':
        return extractor.extract_mean_embedding(audio)
    elif pooling == 'max':
        return extractor.extract_max_embedding(audio)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
