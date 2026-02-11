"""
Recorder Tab Widget - Audio recording functionality.
Extracted from the original audio_recorder.py monolithic file.
"""

import os
import wave
import threading
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QLineEdit, QProgressBar,
    QGroupBox, QCheckBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
import pyaudio


class AudioRecorder(QObject):
    """Handles audio recording operations in a separate thread."""

    recording_complete = pyqtSignal(bool, str)
    level_update = pyqtSignal(int)
    test_complete = pyqtSignal(bool, str)
    countdown_update = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.is_testing = False
        self.frames = []
        self.test_file_path = None

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000
        self.SAMPLE_WIDTH = 2
        
        # Ensure .test directory exists for temporary files
        Path(".test").mkdir(exist_ok=True)

    def get_device_list(self):
        """Get list of all available input devices."""
        devices = []
        try:
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels']
                    })
        except Exception as e:
            print(f"Error getting device list: {e}")
        return devices

    def record_test(self, device_index, duration=3):
        """Record a test audio sample."""
        test_dir = Path(".test")
        test_dir.mkdir(exist_ok=True)
        self.test_file_path = str(test_dir / "test_recording.wav")

        thread = threading.Thread(
            target=self._test_record_thread,
            args=(device_index, duration, self.test_file_path)
        )
        thread.start()

    def _test_record_thread(self, device_index, duration, output_path):
        """Thread function for test recording."""
        try:
            self.is_testing = True
            self.frames = []

            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )

            chunks_needed = int(self.RATE / self.CHUNK * duration)
            chunks_per_second = int(self.RATE / self.CHUNK)

            for i in range(chunks_needed):
                if not self.is_testing:
                    break

                if i % chunks_per_second == 0:
                    remaining = duration - (i // chunks_per_second)
                    self.countdown_update.emit(remaining)

                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    mean_square = np.mean(audio_data.astype(np.float64) ** 2)
                    if mean_square >= 0:
                        rms = np.sqrt(mean_square)
                        level = int((rms / 32768.0) * 100)
                        self.level_update.emit(min(100, level))
                    else:
                        self.level_update.emit(0)
                else:
                    self.level_update.emit(0)

            stream.stop_stream()
            stream.close()

            self._save_wav(output_path)

            self.is_testing = False
            self.test_complete.emit(True, "Test recording complete")

        except Exception as e:
            self.is_testing = False
            self.test_complete.emit(False, "Test failed: " + str(e))

    def play_test(self):
        """Play back the test recording."""
        if not self.test_file_path or not Path(self.test_file_path).exists():
            return False

        thread = threading.Thread(target=self._play_test_thread)
        thread.start()
        return True

    def _play_test_thread(self):
        """Thread function for playing test audio."""
        try:
            wf = wave.open(self.test_file_path, 'rb')

            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True
            )

            data = wf.readframes(self.CHUNK)
            while data:
                stream.write(data)
                data = wf.readframes(self.CHUNK)

            stream.stop_stream()
            stream.close()
            wf.close()

        except Exception:
            pass

    def record(self, device_index, duration, output_path):
        """Record audio to file in a separate thread."""
        thread = threading.Thread(
            target=self._record_thread,
            args=(device_index, duration, output_path)
        )
        thread.start()

    def _record_thread(self, device_index, duration, output_path):
        """Thread function for recording audio."""
        try:
            self.is_recording = True
            self.frames = []

            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )

            chunks_needed = int(self.RATE / self.CHUNK * duration)
            chunks_per_second = int(self.RATE / self.CHUNK)

            for i in range(chunks_needed):
                if not self.is_recording:
                    break

                if i % chunks_per_second == 0:
                    remaining = duration - (i // chunks_per_second)
                    self.countdown_update.emit(remaining)

                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    mean_square = np.mean(audio_data.astype(np.float64) ** 2)
                    if mean_square >= 0:
                        rms = np.sqrt(mean_square)
                        level = int((rms / 32768.0) * 100)
                        self.level_update.emit(min(100, level))
                    else:
                        self.level_update.emit(0)
                else:
                    self.level_update.emit(0)

            stream.stop_stream()
            stream.close()
            
            # Ensure output directory exists before saving
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            self._save_wav(output_path)

            self.is_recording = False
            self.recording_complete.emit(True, f"Saved to {output_path}")

        except Exception as e:
            self.is_recording = False
            self.recording_complete.emit(False, f"Error: {str(e)}")

    def _save_wav(self, filename):
        """Save recorded frames to WAV file."""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.SAMPLE_WIDTH)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def stop_recording(self):
        """Stop current recording."""
        self.is_recording = False

    def cleanup(self):
        """Clean up PyAudio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()


class RecorderWidget(QWidget):
    """Recording tab widget with device selection, testing, and recording."""

    status_message = pyqtSignal(str, int)  # message, timeout_ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recorder = AudioRecorder()
        self.current_index = 1
        self.is_testing = False
        self.is_recording = False
        
        # Batch recording state
        self.batch_remaining = 0
        self.batch_category = None
        self.batch_timer = QTimer()
        self.batch_timer.setSingleShot(True)
        self.batch_timer.timeout.connect(self.start_next_batch_record)

        self.init_ui()
        self.load_devices()

        # Connect signals
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.level_update.connect(self.update_level_meter)
        self.recorder.test_complete.connect(self.on_test_complete)
        self.recorder.countdown_update.connect(self.update_countdown)

    def init_ui(self):
        """Initialize the recording UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Device Selection Group
        device_group = QGroupBox("Audio Device")
        device_layout = QVBoxLayout()

        self.device_combo = QComboBox()
        device_layout.addWidget(QLabel("Microphone:"))
        device_layout.addWidget(self.device_combo)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Level Meter Group
        meter_group = QGroupBox("Audio Test")
        meter_layout = QVBoxLayout()

        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(True)
        self.level_meter.setFormat("%v%")

        test_buttons_layout = QHBoxLayout()

        self.test_record_button = QPushButton("Record Test (3s)")
        self.test_record_button.clicked.connect(self.record_test)

        self.test_play_button = QPushButton("Play Test")
        self.test_play_button.clicked.connect(self.play_test)
        self.test_play_button.setEnabled(False)

        test_buttons_layout.addWidget(self.test_record_button)
        test_buttons_layout.addWidget(self.test_play_button)

        meter_layout.addWidget(self.level_meter)
        meter_layout.addLayout(test_buttons_layout)

        meter_group.setLayout(meter_layout)
        layout.addWidget(meter_group)

        # Configuration Group
        config_group = QGroupBox("Recording Configuration")
        config_layout = QVBoxLayout()

        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (seconds):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(1)
        duration_layout.addWidget(self.duration_spin)
        duration_layout.addStretch()
        config_layout.addLayout(duration_layout)

        # Prefix
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Filename Prefix:"))
        self.prefix_input = QLineEdit("sample")
        prefix_layout.addWidget(self.prefix_input)
        config_layout.addLayout(prefix_layout)

        # Starting Index
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("Starting Index:"))
        self.index_spin = QSpinBox()
        self.index_spin.setRange(1, 999999)
        self.index_spin.setValue(1)
        self.index_spin.valueChanged.connect(self.on_index_changed)
        index_layout.addWidget(self.index_spin)
        index_layout.addStretch()
        config_layout.addLayout(index_layout)
        
        # Batch Mode Controls
        batch_layout = QHBoxLayout()
        
        self.batch_check = QCheckBox("Batch Mode")
        batch_layout.addWidget(self.batch_check)
        
        batch_layout.addWidget(QLabel("Count:"))
        self.batch_count_spin = QSpinBox()
        self.batch_count_spin.setRange(1, 100)
        self.batch_count_spin.setValue(5)
        batch_layout.addWidget(self.batch_count_spin)
        
        batch_layout.addWidget(QLabel("Interval (s):"))
        self.batch_interval_spin = QDoubleSpinBox()
        self.batch_interval_spin.setRange(0.5, 10.0)
        self.batch_interval_spin.setSingleStep(0.5)
        self.batch_interval_spin.setValue(2.0)
        batch_layout.addWidget(self.batch_interval_spin)
        
        batch_layout.addStretch()
        config_layout.addLayout(batch_layout)

        # Next filename display
        self.next_filename_label = QLabel()
        self.update_next_filename_display()
        config_layout.addWidget(self.next_filename_label)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Recording Buttons
        buttons_layout = QHBoxLayout()

        self.record_ok_button = QPushButton("Record OK")
        self.record_ok_button.setMinimumHeight(50)
        self.record_ok_button.clicked.connect(lambda: self.start_recording("OK"))

        self.record_ng_button = QPushButton("Record NG")
        self.record_ng_button.setMinimumHeight(50)
        self.record_ng_button.clicked.connect(lambda: self.start_recording("NG"))

        buttons_layout.addWidget(self.record_ok_button)
        buttons_layout.addWidget(self.record_ng_button)
        
        # Stop Button (initially hidden, used for batch cancellation)
        self.stop_batch_button = QPushButton("Stop Batch")
        self.stop_batch_button.setMinimumHeight(50)
        self.stop_batch_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_batch_button.clicked.connect(self.stop_batch_recording)
        self.stop_batch_button.setVisible(False)
        layout.addWidget(self.stop_batch_button)

        layout.addLayout(buttons_layout)
        layout.addStretch()

    def load_devices(self):
        """Load audio devices into combo box."""
        devices = self.recorder.get_device_list()
        self.device_combo.clear()

        for device in devices:
            self.device_combo.addItem(
                f"{device['name']} ({device['channels']} ch)",
                device['index']
            )

        if not devices:
            self.status_message.emit("No input devices found!", 5000)

    def record_test(self):
        """Record a test audio sample."""
        if self.is_testing or self.is_recording:
            return

        device_index = self.device_combo.currentData()
        if device_index is None:
            self.status_message.emit("No device selected!", 3000)
            return

        self.is_testing = True
        self.test_record_button.setEnabled(False)
        self.test_play_button.setEnabled(False)
        self.device_combo.setEnabled(False)

        self.test_record_button.setText("Recording... (3s)")
        self.status_message.emit("Recording test sample...", 0)

        self.recorder.record_test(device_index, duration=3)

    def play_test(self):
        """Play back the test recording."""
        if self.is_testing or self.is_recording:
            return

        if self.recorder.play_test():
            self.test_play_button.setEnabled(False)
            self.status_message.emit("Playing test recording...", 0)

            QTimer.singleShot(3000, lambda: self.test_play_button.setEnabled(True))
            QTimer.singleShot(3000, lambda: self.status_message.emit("Playback complete", 3000))
        else:
            self.status_message.emit("No test recording available!", 3000)

    def on_test_complete(self, success, message):
        """Handle test recording completion."""
        self.is_testing = False
        self.test_record_button.setEnabled(True)
        self.test_record_button.setText("Record Test (3s)")
        self.device_combo.setEnabled(True)
        self.level_meter.setValue(0)

        if success:
            self.test_play_button.setEnabled(True)
            self.status_message.emit("Test recorded! Click 'Play Test' to listen.", 5000)
        else:
            self.status_message.emit(message, 5000)

    def update_level_meter(self, level):
        """Update level meter."""
        self.level_meter.setValue(level)

    def update_countdown(self, remaining):
        """Update countdown display on buttons."""
        if self.is_testing:
            self.test_record_button.setText(f"Recording... ({remaining}s)")
        elif self.is_recording:
            if "Recording..." in self.record_ok_button.text():
                self.record_ok_button.setText(f"Recording... ({remaining}s)")
            elif "Recording..." in self.record_ng_button.text():
                self.record_ng_button.setText(f"Recording... ({remaining}s)")

    def on_index_changed(self, value):
        """Handle index spin box value change."""
        self.current_index = value
        self.update_next_filename_display()

    def update_next_filename_display(self):
        """Update the display showing the next filename."""
        prefix = self.prefix_input.text() or "sample"
        next_filename = f"{prefix}_{self.current_index}.wav"
        self.next_filename_label.setText(f"<b>Next file:</b> {next_filename}")
        self.next_filename_label.setStyleSheet("color: #00bcd4; background: transparent;")

    def start_recording(self, category):
        """Start recording to OK or NG folder."""
        if self.is_recording:
            return

        device_index = self.device_combo.currentData()
        if device_index is None:
            self.status_message.emit("No device selected!", 3000)
            return

        # Initialize batch mode if checked
        if self.batch_check.isChecked() and self.batch_remaining == 0:
            self.batch_remaining = self.batch_count_spin.value()
            self.batch_category = category
            # We don't use 'start' on timer here, logic flows to record() which triggers update when done.
            # But we need to make sure we treat this as the first of batch.
            self.stop_batch_button.setVisible(True)

        duration = self.duration_spin.value()
        prefix = self.prefix_input.text() or "sample"

        output_dir = Path("output") / category
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{prefix}_{self.current_index}.wav"
        output_path = str(output_dir / filename)

        self.is_recording = True
        self.set_ui_enabled(False)
        self.stop_batch_button.setEnabled(True)

        if category == "OK":
            self.record_ok_button.setText(f"Recording... ({duration}s)")
            self.record_ok_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                }
            """)
        else:
            self.record_ng_button.setText(f"Recording... ({duration}s)")
            self.record_ng_button.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                }
            """)
            
        if self.batch_check.isChecked():
             # Logic fix: if we just started, we decrement for this current one
            if self.batch_remaining > 0:
                 # decrement is done below or here? Let's do it here. 
                 # Because start_recording IS the recording action.
                 self.batch_remaining -= 1

            remaining_msg = f" (Batch: {self.batch_remaining} pending)"
            self.status_message.emit(f"Recording {category}: {filename}{remaining_msg}", 0)
        else:
            self.status_message.emit(f"Recording {category}: {filename}", 0)

        self.recorder.record(device_index, duration, output_path)

    def start_next_batch_record(self):
        """Trigger the next recording in the batch."""
        if self.batch_remaining > 0 and self.batch_category:
            self.start_recording(self.batch_category)
        else:
            self.stop_batch_recording()

    def stop_batch_recording(self):
        """Cancel any ongoing batch recording."""
        self.batch_remaining = 0
        self.batch_category = None
        self.batch_timer.stop()
        self.stop_batch_button.setVisible(False)
        
        # Stop actual recording if currently recording
        if self.is_recording:
             self.recorder.stop_recording()
             
        self.status_message.emit("Batch recording stopped.", 3000)
        self.set_ui_enabled(True) # Ensure UI comes back if we stopped while waiting

    def on_recording_complete(self, success, message):
        """Handle recording completion."""
        self.is_recording = False

        self.record_ok_button.setText("Record OK")
        self.record_ng_button.setText("Record NG")
        self.record_ok_button.setStyleSheet("")
        self.record_ng_button.setStyleSheet("")

        if success:
            self.current_index += 1
            self.index_spin.setValue(self.current_index)
            
            # Check if we should continue batch
            if self.batch_remaining > 0:
                interval = self.batch_interval_spin.value()
                self.status_message.emit(f"Saved. Next recording in {interval}s...", int(interval * 1000))
                
                # Keep UI disabled, start timer
                self.batch_timer.start(int(interval * 1000))
                return
            else:
                 # Batch finished or just single record finished
                self.status_message.emit(message, 5000)
        else:
            self.status_message.emit(f"Recording failed: {message}", 5000)
            self.batch_remaining = 0

        self.level_meter.setValue(0)
        self.set_ui_enabled(True)
        self.stop_batch_button.setVisible(False)

    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements."""
        self.device_combo.setEnabled(enabled)
        self.test_record_button.setEnabled(enabled)
        self.duration_spin.setEnabled(enabled)
        self.prefix_input.setEnabled(enabled)
        self.index_spin.setEnabled(enabled)
        self.record_ok_button.setEnabled(enabled)
        self.record_ng_button.setEnabled(enabled)
        
        # We can enable batch controls only when not recording at all
        self.batch_check.setEnabled(enabled)
        self.batch_count_spin.setEnabled(enabled)
        self.batch_interval_spin.setEnabled(enabled)

    def cleanup(self):
        """Clean up resources."""
        self.batch_timer.stop()
        self.recorder.cleanup()
