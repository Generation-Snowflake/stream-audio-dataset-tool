"""
Custom classifier model for OK/NG sound classification.
Uses transfer learning with frozen YAMNet embeddings.
Adapted from yam-sound-dectection-pipeline.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_classifier_model(input_dim: int = 1024,
                           hidden_units: int = 256,
                           dropout_rate: float = 0.3,
                           num_classes: int = 2) -> Model:
    """
    Build custom classifier head for YAMNet embeddings.

    Architecture:
        Input (1024) -> Dense(256, ReLU) -> BN -> Dropout -> Dense(128, ReLU) -> Dense(2, Softmax)

    Args:
        input_dim: Dimension of YAMNet embeddings (default: 1024)
        hidden_units: Number of units in hidden layer (default: 256)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        num_classes: Number of output classes (default: 2 for OK/NG)

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(input_dim,), name='yamnet_embeddings')

    x = layers.Dense(hidden_units, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization(name='batch_norm')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='sound_classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info("✓ Model built and compiled successfully")
    return model


def create_callbacks(checkpoint_dir: str = 'models/checkpoints',
                     patience: int = 10) -> list:
    """
    Create training callbacks.

    Args:
        checkpoint_dir: Directory to save model checkpoints
        patience: Early stopping patience (epochs)

    Returns:
        List of Keras callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]

    return callbacks


def evaluate_model(model: Model,
                   X_test: tf.Tensor,
                   y_test: tf.Tensor) -> dict:
    """
    Evaluate model performance with detailed metrics.

    Args:
        model: Trained Keras model
        X_test: Test embeddings
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    class_names = ['OK', 'NG']
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'classification_report': report,
        'confusion_matrix': cm
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"{'='*50}\n")

    return results


def plot_training_history(history: keras.callbacks.History,
                          save_path: str = 'training_history.png'):
    """
    Plot training history (accuracy and loss curves).

    Args:
        history: Keras training history object
        save_path: Path to save plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Training history plot saved to {save_path}")
    plt.close()
