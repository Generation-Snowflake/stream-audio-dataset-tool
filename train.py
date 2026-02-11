#!/usr/bin/env python3
"""
Training script for YAMNet sound classification model.

Usage:
    python train.py --data_dir output --epochs 50 --batch_size 16
    python train.py --data_dir output --epochs 10 --batch_size 8 --output_dir models/my_model.keras
"""

import argparse
import os
import numpy as np
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from utils import (
    load_dataset,
    get_class_weights,
    create_embedding_dataset,
    YAMNetFeatureExtractor
)
from models import (
    build_classifier_model,
    create_callbacks,
    evaluate_model,
    plot_training_history
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YAMNet-based sound classifier for OK/NG detection'
    )

    parser.add_argument(
        '--data_dir', type=str, default='output',
        help='Path to training data directory with OK/ and NG/ subdirs (default: output)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for training (default: 16)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='models/sound_classifier.keras',
        help='Path to save trained model (default: models/sound_classifier.keras)'
    )
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--hidden_units', type=int, default=256,
        help='Number of hidden units in classifier (default: 256)'
    )
    parser.add_argument(
        '--dropout_rate', type=float, default=0.3,
        help='Dropout rate for regularization (default: 0.3)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("YAMNet Sound Classification Training")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Validation split: {args.val_split}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✓ GPU available: {len(gpus)} device(s)")
    else:
        logger.info("Running on CPU")

    # Step 1: Load dataset
    logger.info("\n[Step 1/5] Loading dataset...")
    train_files, train_labels, val_files, val_labels = load_dataset(
        args.data_dir, val_split=args.val_split
    )

    # Step 2: Initialize YAMNet
    logger.info("\n[Step 2/5] Loading YAMNet model...")
    feature_extractor = YAMNetFeatureExtractor()

    # Step 3: Extract embeddings
    logger.info("\n[Step 3/5] Extracting embeddings...")

    logger.info("Extracting training embeddings...")
    X_train, y_train = create_embedding_dataset(
        train_files, train_labels, feature_extractor
    )

    X_val, y_val = None, None
    if len(val_files) > 0:
        logger.info("Extracting validation embeddings...")
        X_val, y_val = create_embedding_dataset(
            val_files, val_labels, feature_extractor
        )

    class_weights = get_class_weights(train_labels)

    # Step 4: Build model
    logger.info("\n[Step 4/5] Building classifier model...")
    model = build_classifier_model(
        input_dim=1024,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        num_classes=2
    )
    model.summary()

    # Step 5: Train
    logger.info("\n[Step 5/5] Training model...")
    callbacks = create_callbacks(checkpoint_dir='models/checkpoints', patience=10)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    if X_val is not None:
        logger.info("\n[Evaluation] Evaluating model on validation set...")
        evaluate_model(model, X_val, y_val)

    # Plot training history
    logger.info("\n[Visualization] Generating training plots...")
    plot_training_history(history, save_path='training_history.png')

    # Save model
    logger.info(f"\n[Saving] Saving model to {args.output_dir}...")
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    model.save(args.output_dir)
    logger.info(f"✓ Model saved successfully to {args.output_dir}")

    # Save metadata
    import json
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'num_train_samples': len(train_files),
        'num_val_samples': len(val_files),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_units': args.hidden_units,
        'dropout_rate': args.dropout_rate,
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None
    }

    metadata_path = os.path.join(os.path.dirname(args.output_dir), 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Training metadata saved to {metadata_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
