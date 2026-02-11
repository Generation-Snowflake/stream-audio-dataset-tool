"""
Training Tab Widget - Model training interface.
Provides GUI for training YAMNet-based sound classifiers.
"""

import os
import threading
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFileDialog,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingWidget(QWidget):
    """Training tab widget for model training."""

    status_message = pyqtSignal(str, int)
    _log_signal = pyqtSignal(str)
    _progress_signal = pyqtSignal(int)
    _training_done_signal = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_training = False
        self.training_history = None
        self.init_ui()

        # Connect internal signals for thread-safe UI updates
        self._log_signal.connect(self._append_log)
        self._progress_signal.connect(self._update_progress)
        self._training_done_signal.connect(self._on_training_done)

    def init_ui(self):
        """Initialize the training UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Data directory selection
        data_group = QGroupBox("Training Data")
        data_layout = QHBoxLayout()

        self.data_label = QLabel("Default: output/")
        self.data_label.setStyleSheet("color: #b0b0b0; background: transparent;")
        data_layout.addWidget(self.data_label, 1)

        self.data_path = os.path.join(os.getcwd(), "output")

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(120)
        browse_btn.clicked.connect(self.browse_data)
        data_layout.addWidget(browse_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Hyperparameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()

        # Epochs
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        row1.addWidget(self.epochs_spin)
        row1.addStretch()
        params_layout.addLayout(row1)

        # Batch size
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        row2.addWidget(self.batch_spin)
        row2.addStretch()
        params_layout.addLayout(row2)

        # Hidden units
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Hidden Units:"))
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(32, 1024)
        self.hidden_spin.setValue(256)
        row3.addWidget(self.hidden_spin)
        row3.addStretch()
        params_layout.addLayout(row3)

        # Dropout
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Dropout Rate:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setValue(0.3)
        self.dropout_spin.setSingleStep(0.05)
        row4.addWidget(self.dropout_spin)
        row4.addStretch()
        params_layout.addLayout(row4)

        # Validation split
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Validation Split:"))
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setValue(0.2)
        self.val_split_spin.setSingleStep(0.05)
        row5.addWidget(self.val_split_spin)
        row5.addStretch()
        params_layout.addLayout(row5)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Model output
        output_group = QGroupBox("Model Output")
        output_layout = QHBoxLayout()

        self.model_output_label = QLabel("models/sound_classifier.keras")
        self.model_output_label.setStyleSheet("color: #b0b0b0; background: transparent;")
        output_layout.addWidget(self.model_output_label, 1)

        self.model_output_path = "models/sound_classifier.keras"

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Training controls
        controls_layout = QHBoxLayout()

        self.train_button = QPushButton("🚀 Start Training")
        self.train_button.setMinimumHeight(50)
        self.train_button.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_button)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        controls_layout.addWidget(self.stop_button)

        layout.addLayout(controls_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px;
                font-family: monospace; font-size: 10pt;
            }
        """)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def browse_data(self):
        """Browse for training data directory."""
        default_dir = os.path.join(os.getcwd(), "output")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Training Data Directory", default_dir
        )
        if folder:
            self.data_path = folder
            self.data_label.setText(folder)

    def _append_log(self, text):
        """Append text to training log (thread-safe)."""
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_progress(self, value):
        """Update progress bar (thread-safe)."""
        self.progress_bar.setValue(value)

    def _on_training_done(self, success, message):
        """Handle training completion (thread-safe)."""
        self.is_training = False
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if success:
            self.status_message.emit("Training completed successfully!", 5000)
        else:
            self.status_message.emit(f"Training failed: {message}", 5000)

    def start_training(self):
        """Start model training in a background thread."""
        if self.is_training:
            return

        # Validate data directory
        ok_dir = os.path.join(self.data_path, "OK")
        ng_dir = os.path.join(self.data_path, "NG")

        if not os.path.exists(ok_dir) and not os.path.exists(ng_dir):
            self.status_message.emit("No OK/ or NG/ subdirectories found in data directory!", 5000)
            return

        self.is_training = True
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_text.clear()
        self.progress_bar.setValue(0)

        thread = threading.Thread(target=self._train_thread, daemon=True)
        thread.start()

    def stop_training(self):
        """Stop training."""
        self.is_training = False
        self._log_signal.emit("⏹ Training stop requested...")

    def _train_thread(self):
        """Background thread for model training."""
        try:
            self._log_signal.emit("=" * 50)
            self._log_signal.emit("🚀 Starting YAMNet Sound Classification Training")
            self._log_signal.emit("=" * 50)

            # Step 1: Load dataset
            self._log_signal.emit("\n[Step 1/5] Loading dataset...")
            self._progress_signal.emit(5)

            from utils.data_loader import load_dataset, get_class_weights, create_embedding_dataset

            train_files, train_labels, val_files, val_labels = load_dataset(
                self.data_path,
                val_split=self.val_split_spin.value()
            )

            self._log_signal.emit(f"  Train files: {len(train_files)}")
            self._log_signal.emit(f"  Validation files: {len(val_files)}")

            if not self.is_training:
                self._training_done_signal.emit(False, "Stopped by user")
                return

            # Step 2: Load YAMNet
            self._log_signal.emit("\n[Step 2/5] Loading YAMNet model...")
            self._progress_signal.emit(15)

            from utils.feature_extraction import YAMNetFeatureExtractor
            feature_extractor = YAMNetFeatureExtractor()

            if not self.is_training:
                self._training_done_signal.emit(False, "Stopped by user")
                return

            # Step 3: Extract embeddings
            self._log_signal.emit("\n[Step 3/5] Extracting embeddings...")
            self._progress_signal.emit(25)

            X_train, y_train = create_embedding_dataset(
                train_files, train_labels, feature_extractor
            )

            self._log_signal.emit(f"  Train embeddings shape: {X_train.shape}")

            X_val, y_val = None, None
            if len(val_files) > 0:
                X_val, y_val = create_embedding_dataset(
                    val_files, val_labels, feature_extractor
                )
                self._log_signal.emit(f"  Val embeddings shape: {X_val.shape}")

            class_weights = get_class_weights(train_labels)
            self._log_signal.emit(f"  Class weights: {class_weights}")

            if not self.is_training:
                self._training_done_signal.emit(False, "Stopped by user")
                return

            # Step 4: Build model
            self._log_signal.emit("\n[Step 4/5] Building classifier model...")
            self._progress_signal.emit(40)

            from models.classifier import build_classifier_model, create_callbacks

            model = build_classifier_model(
                input_dim=1024,
                hidden_units=self.hidden_spin.value(),
                dropout_rate=self.dropout_spin.value(),
                num_classes=2
            )

            # Log model summary
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            for line in summary_lines:
                self._log_signal.emit(f"  {line}")

            # Step 5: Train
            self._log_signal.emit("\n[Step 5/5] Training model...")
            self._progress_signal.emit(50)

            callbacks = create_callbacks(
                checkpoint_dir='models/checkpoints',
                patience=10
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_spin.value(),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )

            self.training_history = history

            # Log training results
            for epoch in range(len(history.history['accuracy'])):
                acc = history.history['accuracy'][epoch]
                loss = history.history['loss'][epoch]
                msg = f"  Epoch {epoch+1}: acc={acc:.4f}, loss={loss:.4f}"
                if 'val_accuracy' in history.history:
                    val_acc = history.history['val_accuracy'][epoch]
                    val_loss = history.history['val_loss'][epoch]
                    msg += f", val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
                self._log_signal.emit(msg)

            self._progress_signal.emit(85)

            # Evaluate
            if X_val is not None:
                self._log_signal.emit("\n[Evaluation]")
                from models.classifier import evaluate_model
                results = evaluate_model(model, X_val, y_val)
                self._log_signal.emit(f"  Test Accuracy: {results['test_accuracy']:.4f}")
                self._log_signal.emit(f"  Test Loss: {results['test_loss']:.4f}")

            # Plot history
            self._log_signal.emit("\n[Saving training plot]")
            from models.classifier import plot_training_history
            plot_training_history(history, save_path='training_history.png')
            self._log_signal.emit("  ✓ Saved to training_history.png")

            # Save model
            self._log_signal.emit(f"\n[Saving model to {self.model_output_path}]")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            model.save(self.model_output_path)
            self._log_signal.emit(f"  ✓ Model saved successfully")

            # Save metadata
            import json
            from datetime import datetime
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'data_dir': self.data_path,
                'num_train_samples': len(train_files),
                'num_val_samples': len(val_files),
                'epochs': self.epochs_spin.value(),
                'batch_size': self.batch_spin.value(),
                'hidden_units': self.hidden_spin.value(),
                'dropout_rate': self.dropout_spin.value(),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None
            }

            metadata_path = os.path.join(os.path.dirname(self.model_output_path), 'training_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self._log_signal.emit(f"  ✓ Metadata saved to {metadata_path}")

            self._progress_signal.emit(100)
            self._log_signal.emit("\n" + "=" * 50)
            self._log_signal.emit("✅ Training completed successfully!")
            self._log_signal.emit("=" * 50)

            self._training_done_signal.emit(True, "")

        except Exception as e:
            self._log_signal.emit(f"\n❌ Error: {str(e)}")
            import traceback
            self._log_signal.emit(traceback.format_exc())
            self._training_done_signal.emit(False, str(e))
