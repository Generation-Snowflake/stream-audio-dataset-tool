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

try:
    from utils.feature_extraction import (
        YAMNetFeatureExtractor,
        extract_embeddings_from_file
    )
except ImportError:
    YAMNetFeatureExtractor = None
    extract_embeddings_from_file = None

try:
    from utils.data_loader import (
        load_dataset,
        get_class_weights,
        create_embedding_dataset,
        augment_audio
    )
except ImportError:
    load_dataset = None
    get_class_weights = None
    create_embedding_dataset = None
    augment_audio = None

from utils.motor_controller import MotorController

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
    # Motor control
    'MotorController',
]
