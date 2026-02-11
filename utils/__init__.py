"""Utilities package for Audio Dataset Tool."""

from utils.audio_preprocessing import (
    load_and_preprocess_audio,
    resample_audio,
    normalize_audio,
    validate_audio_format,
    pad_or_trim_audio,
    apply_gain
)

from utils.audio_analysis import (
    mel_spectrogram,
    avg_energy,
    temporal_variance,
    load_audio_for_analysis
)

from utils.feature_extraction import (
    YAMNetFeatureExtractor,
    extract_embeddings_from_file
)

from utils.data_loader import (
    load_dataset,
    get_class_weights,
    create_embedding_dataset,
    augment_audio
)

__all__ = [
    # Audio preprocessing
    'load_and_preprocess_audio',
    'resample_audio',
    'normalize_audio',
    'validate_audio_format',
    'pad_or_trim_audio',
    'apply_gain',
    # Audio analysis
    'mel_spectrogram',
    'avg_energy',
    'temporal_variance',
    'load_audio_for_analysis',
    # Feature extraction
    'YAMNetFeatureExtractor',
    'extract_embeddings_from_file',
    # Data loading
    'load_dataset',
    'get_class_weights',
    'create_embedding_dataset',
    'augment_audio',
]
