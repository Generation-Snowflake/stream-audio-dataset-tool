"""
Inference Tab Widget - Model testing and prediction interface.
Supports single file, batch, and live microphone prediction.
"""

import os
import threading
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFileDialog,
    QTextEdit, QProgressBar, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class InferenceWidget(QWidget):
    """Inference tab widget for model testing."""

    status_message = pyqtSignal(str, int)
    _log_signal = pyqtSignal(str)
    _progress_signal = pyqtSignal(int)
    _result_signal = pyqtSignal(str, float, float, float)  # label, confidence, ok_prob, ng_prob
    _done_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.feature_extractor = None
        self.model_path = "models/sound_classifier.keras"
        self.init_ui()

        # Connect internal signals
        self._log_signal.connect(self._append_log)
        self._progress_signal.connect(self._update_progress)
        self._result_signal.connect(self._show_result)
        self._done_signal.connect(self._on_done)

    def init_ui(self):
        """Initialize the inference UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Model loading
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()

        self.model_label = QLabel("models/sound_classifier.keras")
        self.model_label.setStyleSheet("color: #b0b0b0; background: transparent;")
        model_layout.addWidget(self.model_label, 1)

        browse_model_btn = QPushButton("Browse Model...")
        browse_model_btn.setMaximumWidth(140)
        browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_model_btn)

        load_model_btn = QPushButton("Load Model")
        load_model_btn.setMaximumWidth(120)
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Model status
        self.model_status_label = QLabel("⚠️ No model loaded")
        self.model_status_label.setStyleSheet("color: #f9e2af; background: transparent; padding: 5px;")
        layout.addWidget(self.model_status_label)

        # Prediction mode selection
        mode_group = QGroupBox("Prediction Mode")
        mode_layout = QVBoxLayout()

        # Single file
        single_layout = QHBoxLayout()
        self.predict_single_btn = QPushButton("📄 Predict Single File")
        self.predict_single_btn.setMinimumHeight(45)
        self.predict_single_btn.clicked.connect(self.predict_single)
        self.predict_single_btn.setEnabled(False)
        single_layout.addWidget(self.predict_single_btn)
        mode_layout.addLayout(single_layout)

        # Batch prediction
        batch_layout = QHBoxLayout()
        self.predict_batch_btn = QPushButton("📁 Predict Batch (Folder)")
        self.predict_batch_btn.setMinimumHeight(45)
        self.predict_batch_btn.clicked.connect(self.predict_batch)
        self.predict_batch_btn.setEnabled(False)
        batch_layout.addWidget(self.predict_batch_btn)
        mode_layout.addLayout(batch_layout)

        # Export results
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("💾 Export Results to CSV")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        mode_layout.addLayout(export_layout)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Result display
        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout()

        self.result_label = QLabel("--")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 32pt; font-weight: bold; color: #6c7086;
            background: transparent; padding: 15px;
        """)
        result_layout.addWidget(self.result_label)

        # Probability bars
        prob_layout = QHBoxLayout()

        ok_layout = QVBoxLayout()
        ok_layout.addWidget(QLabel("OK Probability:"))
        self.ok_bar = QProgressBar()
        self.ok_bar.setRange(0, 100)
        self.ok_bar.setValue(0)
        self.ok_bar.setFormat("%v%")
        self.ok_bar.setStyleSheet("""
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }
        """)
        ok_layout.addWidget(self.ok_bar)
        prob_layout.addLayout(ok_layout)

        ng_layout = QVBoxLayout()
        ng_layout.addWidget(QLabel("NG Probability:"))
        self.ng_bar = QProgressBar()
        self.ng_bar.setRange(0, 100)
        self.ng_bar.setValue(0)
        self.ng_bar.setFormat("%v%")
        self.ng_bar.setStyleSheet("""
            QProgressBar::chunk { background-color: #f38ba8; border-radius: 3px; }
        """)
        ng_layout.addWidget(self.ng_bar)
        prob_layout.addLayout(ng_layout)

        result_layout.addLayout(prob_layout)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log
        log_group = QGroupBox("Inference Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
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

        self.batch_results = []

    def _append_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _show_result(self, label, confidence, ok_prob, ng_prob):
        """Display prediction result."""
        ok_pct = int(ok_prob * 100)
        ng_pct = int(ng_prob * 100)

        self.ok_bar.setValue(ok_pct)
        self.ng_bar.setValue(ng_pct)

        if label == 'OK':
            self.result_label.setText(f"✅ OK ({confidence*100:.1f}%)")
            self.result_label.setStyleSheet("""
                font-size: 32pt; font-weight: bold; color: #a6e3a1;
                background: transparent; padding: 15px;
            """)
        else:
            self.result_label.setText(f"⚠️ NG ({confidence*100:.1f}%)")
            self.result_label.setStyleSheet("""
                font-size: 32pt; font-weight: bold; color: #f38ba8;
                background: transparent; padding: 15px;
            """)

    def _on_done(self):
        self.predict_single_btn.setEnabled(True)
        self.predict_batch_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def browse_model(self):
        """Browse for a model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File",
            os.path.join(os.getcwd(), "models"),
            "Keras Model (*.keras);;All Files (*)"
        )
        if file_path:
            self.model_path = file_path
            self.model_label.setText(file_path)

    def load_model(self):
        """Load the selected model."""
        if not os.path.exists(self.model_path):
            self.status_message.emit(f"Model not found: {self.model_path}", 5000)
            return

        self.model_status_label.setText("⏳ Loading model...")
        self.model_status_label.setStyleSheet("color: #f9e2af; background: transparent; padding: 5px;")

        thread = threading.Thread(target=self._load_model_thread, daemon=True)
        thread.start()

    def _load_model_thread(self):
        """Load model in background thread."""
        try:
            from tensorflow import keras
            from utils.feature_extraction import YAMNetFeatureExtractor

            self._log_signal.emit("Loading model...")
            self.model = keras.models.load_model(self.model_path)
            self._log_signal.emit("✓ Model loaded")

            self._log_signal.emit("Loading YAMNet feature extractor...")
            self.feature_extractor = YAMNetFeatureExtractor()
            self._log_signal.emit("✓ YAMNet loaded")

            self.model_status_label.setText("✅ Model loaded successfully")
            self.model_status_label.setStyleSheet("color: #a6e3a1; background: transparent; padding: 5px;")

            self.predict_single_btn.setEnabled(True)
            self.predict_batch_btn.setEnabled(True)

            self.status_message.emit("Model loaded successfully!", 3000)

        except Exception as e:
            self._log_signal.emit(f"❌ Error loading model: {str(e)}")
            self.model_status_label.setText(f"❌ Failed to load model")
            self.model_status_label.setStyleSheet("color: #f38ba8; background: transparent; padding: 5px;")
            self.status_message.emit(f"Failed to load model: {str(e)}", 5000)

    def predict_single(self):
        """Predict a single audio file."""
        if self.model is None:
            self.status_message.emit("Please load a model first!", 3000)
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File",
            os.path.join(os.getcwd(), "output"),
            "WAV Files (*.wav);;All Files (*)"
        )
        if not file_path:
            return

        self.predict_single_btn.setEnabled(False)
        thread = threading.Thread(target=self._predict_single_thread, args=(file_path,), daemon=True)
        thread.start()

    def _predict_single_thread(self, file_path):
        """Predict single file in background."""
        try:
            from utils.audio_preprocessing import load_and_preprocess_audio

            self._log_signal.emit(f"\nPredicting: {os.path.basename(file_path)}")

            audio = load_and_preprocess_audio(file_path)
            embedding = self.feature_extractor.extract_mean_embedding(audio)
            embedding = np.expand_dims(embedding, axis=0)

            predictions = self.model.predict(embedding, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])

            class_names = ['OK', 'NG']
            class_label = class_names[class_idx]

            ok_prob = float(predictions[0][0])
            ng_prob = float(predictions[0][1])

            self._log_signal.emit(f"  Result: {class_label} (confidence: {confidence*100:.2f}%)")
            self._log_signal.emit(f"  OK: {ok_prob*100:.2f}% | NG: {ng_prob*100:.2f}%")

            self._result_signal.emit(class_label, confidence, ok_prob, ng_prob)

            self.batch_results.append({
                'filename': os.path.basename(file_path),
                'prediction': class_label,
                'confidence': confidence,
                'ok_probability': ok_prob,
                'ng_probability': ng_prob
            })
            self.export_btn.setEnabled(True)

        except Exception as e:
            self._log_signal.emit(f"❌ Error: {str(e)}")

        self._done_signal.emit()

    def predict_batch(self):
        """Predict all files in a directory."""
        if self.model is None:
            self.status_message.emit("Please load a model first!", 3000)
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select Audio Directory",
            os.path.join(os.getcwd(), "output")
        )
        if not folder:
            return

        self.predict_single_btn.setEnabled(False)
        self.predict_batch_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.batch_results = []

        thread = threading.Thread(target=self._predict_batch_thread, args=(folder,), daemon=True)
        thread.start()

    def _predict_batch_thread(self, folder):
        """Predict batch in background."""
        try:
            from utils.audio_preprocessing import load_and_preprocess_audio

            wav_files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith('.wav')
            ])

            if not wav_files:
                self._log_signal.emit("No .wav files found in directory")
                self._done_signal.emit()
                return

            self._log_signal.emit(f"\n{'='*40}")
            self._log_signal.emit(f"Batch prediction: {len(wav_files)} files")
            self._log_signal.emit(f"{'='*40}\n")

            ok_count = 0
            ng_count = 0

            for i, filename in enumerate(wav_files):
                file_path = os.path.join(folder, filename)

                try:
                    audio = load_and_preprocess_audio(file_path)
                    embedding = self.feature_extractor.extract_mean_embedding(audio)
                    embedding = np.expand_dims(embedding, axis=0)

                    predictions = self.model.predict(embedding, verbose=0)
                    class_idx = np.argmax(predictions[0])
                    confidence = float(predictions[0][class_idx])
                    class_names = ['OK', 'NG']
                    class_label = class_names[class_idx]

                    ok_prob = float(predictions[0][0])
                    ng_prob = float(predictions[0][1])

                    if class_label == 'OK':
                        ok_count += 1
                    else:
                        ng_count += 1

                    self.batch_results.append({
                        'filename': filename,
                        'prediction': class_label,
                        'confidence': confidence,
                        'ok_probability': ok_prob,
                        'ng_probability': ng_prob
                    })

                    icon = "✅" if class_label == 'OK' else "⚠️"
                    self._log_signal.emit(
                        f"  {icon} [{i+1}/{len(wav_files)}] {filename}: "
                        f"{class_label} ({confidence*100:.1f}%)"
                    )

                except Exception as e:
                    self._log_signal.emit(f"  ❌ [{i+1}/{len(wav_files)}] {filename}: ERROR - {str(e)}")

                self._progress_signal.emit(int((i + 1) / len(wav_files) * 100))

            # Summary
            self._log_signal.emit(f"\n{'='*40}")
            self._log_signal.emit(f"Summary: {ok_count} OK, {ng_count} NG out of {len(wav_files)} files")
            avg_conf = np.mean([r['confidence'] for r in self.batch_results]) if self.batch_results else 0
            self._log_signal.emit(f"Average confidence: {avg_conf*100:.2f}%")
            self._log_signal.emit(f"{'='*40}")

            self.export_btn.setEnabled(True)
            self.status_message.emit(
                f"Batch prediction done: {ok_count} OK, {ng_count} NG", 5000
            )

        except Exception as e:
            self._log_signal.emit(f"❌ Batch error: {str(e)}")

        self._done_signal.emit()

    def export_results(self):
        """Export batch results to CSV."""
        if not self.batch_results:
            self.status_message.emit("No results to export!", 3000)
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results",
            os.path.join(os.getcwd(), "inference_results.csv"),
            "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'filename', 'prediction', 'confidence',
                    'ok_probability', 'ng_probability'
                ])
                writer.writeheader()
                writer.writerows(self.batch_results)

            self._log_signal.emit(f"\n✓ Results exported to {file_path}")
            self.status_message.emit(f"Results exported to {file_path}", 5000)

        except Exception as e:
            self.status_message.emit(f"Export failed: {str(e)}", 5000)
