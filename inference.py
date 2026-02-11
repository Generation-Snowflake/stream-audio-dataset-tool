#!/usr/bin/env python3
"""
Inference script for YAMNet sound classification.

Usage:
    # Single file prediction
    python inference.py --audio output/OK/sample_1.wav --model models/sound_classifier.keras

    # Batch prediction
    python inference.py --audio_dir output/OK --model models/sound_classifier.keras

    # Batch with CSV output
    python inference.py --audio_dir output/OK --model models/sound_classifier.keras --output_csv results.csv
"""

import argparse
import os
import numpy as np
import logging
from typing import Tuple

from tensorflow import keras

from utils import (
    load_and_preprocess_audio,
    YAMNetFeatureExtractor,
    validate_audio_format
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_single(audio_path: str,
                   model_path: str,
                   feature_extractor: YAMNetFeatureExtractor = None) -> Tuple[str, float]:
    """
    Predict class for a single audio file.

    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        feature_extractor: YAMNetFeatureExtractor instance (optional)

    Returns:
        Tuple of (class_label, confidence_score)
    """
    if not validate_audio_format(audio_path):
        raise ValueError(f"Invalid audio file: {audio_path}")

    logger.info(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    if feature_extractor is None:
        feature_extractor = YAMNetFeatureExtractor()

    logger.info(f"Processing audio file: {audio_path}")
    audio = load_and_preprocess_audio(audio_path)

    embedding = feature_extractor.extract_mean_embedding(audio)
    embedding = np.expand_dims(embedding, axis=0)

    predictions = model.predict(embedding, verbose=0)

    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    class_names = ['OK', 'NG']
    class_label = class_names[class_idx]

    return class_label, float(confidence)


def predict_batch(audio_dir: str, model_path: str, output_csv: str = None) -> list:
    """
    Predict classes for all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        model_path: Path to trained model
        output_csv: Optional path to save results as CSV

    Returns:
        List of prediction dictionaries
    """
    audio_files = sorted([
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(('.wav', '.mp3', '.flac'))
    ])

    if len(audio_files) == 0:
        logger.warning(f"No audio files found in {audio_dir}")
        return []

    logger.info(f"Found {len(audio_files)} audio files")

    logger.info(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    feature_extractor = YAMNetFeatureExtractor()

    results = []

    for idx, filename in enumerate(audio_files):
        audio_path = os.path.join(audio_dir, filename)

        try:
            audio = load_and_preprocess_audio(audio_path)
            embedding = feature_extractor.extract_mean_embedding(audio)
            embedding = np.expand_dims(embedding, axis=0)

            predictions = model.predict(embedding, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]

            class_names = ['OK', 'NG']
            class_label = class_names[class_idx]

            result = {
                'filename': filename,
                'prediction': class_label,
                'confidence': float(confidence),
                'ok_probability': float(predictions[0][0]),
                'ng_probability': float(predictions[0][1])
            }

            results.append(result)
            logger.info(f"[{idx+1}/{len(audio_files)}] {filename}: {class_label} ({confidence:.3f})")

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue

    # Save CSV
    if output_csv and results:
        import csv
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"✓ Results saved to {output_csv}")

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YAMNet sound classification inference'
    )
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--audio_dir', type=str, help='Path to directory for batch prediction')
    parser.add_argument('--model', type=str, default='models/sound_classifier.keras',
                        help='Path to trained model (default: models/sound_classifier.keras)')
    parser.add_argument('--output_csv', type=str, help='Path to save CSV results')

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    if not args.audio and not args.audio_dir:
        logger.error("Please provide either --audio or --audio_dir")
        return

    if args.audio and args.audio_dir:
        logger.error("Please provide only one of --audio or --audio_dir")
        return

    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return

    logger.info("=" * 60)
    logger.info("YAMNet Sound Classification Inference")
    logger.info("=" * 60)

    try:
        if args.audio:
            class_label, confidence = predict_single(args.audio, args.model)

            logger.info("\n" + "=" * 60)
            logger.info(f"Prediction: {class_label}")
            logger.info(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            logger.info("=" * 60)

        elif args.audio_dir:
            results = predict_batch(args.audio_dir, args.model, output_csv=args.output_csv)

            logger.info(f"\n✓ Processed {len(results)} files")

            ok_count = sum(1 for r in results if r['prediction'] == 'OK')
            ng_count = sum(1 for r in results if r['prediction'] == 'NG')
            avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0

            logger.info("\nSummary:")
            logger.info(f"  OK predictions: {ok_count}")
            logger.info(f"  NG predictions: {ng_count}")
            logger.info(f"  Average confidence: {avg_confidence:.4f}")

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


if __name__ == '__main__':
    main()
