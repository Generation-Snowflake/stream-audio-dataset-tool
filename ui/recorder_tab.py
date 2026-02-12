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
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QPushButton, QComboBox, QLabel, QSpinBox, QLineEdit, QProgressBar,
    QGroupBox, QCheckBox, QDoubleSpinBox, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
import pyaudio
from utils.motor_controller import MotorController


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
        self.motor = MotorController()
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

        # Motor signals
        self.motor.cycle_complete.connect(self.on_motor_cycle_complete)
        self.motor.status_update.connect(self.on_motor_status)
        self.motor.error.connect(self.on_motor_error)

    def init_ui(self):
        """Initialize the recording UI."""
        # Use a scroll area so everything fits on smaller screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
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

        # ── Motor Control Group ──
        motor_group = QGroupBox("⚙️ Motor Control (Arduino)")
        motor_main_layout = QVBoxLayout()

        # Enable checkbox
        self.motor_enable_check = QCheckBox("Enable Motor Control")
        motor_main_layout.addWidget(self.motor_enable_check)

        # Connection row
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("Serial Port:"))
        self.serial_port_combo = QComboBox()
        self.serial_port_combo.setMinimumWidth(160)
        conn_layout.addWidget(self.serial_port_combo)

        self.refresh_ports_button = QPushButton("🔄")
        self.refresh_ports_button.setFixedWidth(36)
        self.refresh_ports_button.setToolTip("Refresh serial ports")
        self.refresh_ports_button.clicked.connect(self.refresh_serial_ports)
        conn_layout.addWidget(self.refresh_ports_button)

        self.connect_motor_button = QPushButton("Connect")
        self.connect_motor_button.clicked.connect(self.toggle_motor_connection)
        conn_layout.addWidget(self.connect_motor_button)
        conn_layout.addStretch()
        motor_main_layout.addLayout(conn_layout)

        # Settings grid
        settings_grid = QGridLayout()
        settings_grid.setHorizontalSpacing(12)
        settings_grid.setVerticalSpacing(6)

        # Row 0: Baud Rate
        settings_grid.addWidget(QLabel("Baud Rate:"), 0, 0)
        self.baud_rate_spin = QSpinBox()
        self.baud_rate_spin.setRange(9600, 2000000)
        self.baud_rate_spin.setValue(115200)
        self.baud_rate_spin.setSingleStep(9600)
        settings_grid.addWidget(self.baud_rate_spin, 0, 1)

        # Row 0: Microsteps/Rev
        settings_grid.addWidget(QLabel("Microsteps/Rev:"), 0, 2)
        self.microsteps_spin = QSpinBox()
        self.microsteps_spin.setRange(200, 25600)
        self.microsteps_spin.setValue(1600)
        self.microsteps_spin.setSingleStep(200)
        settings_grid.addWidget(self.microsteps_spin, 0, 3)

        # Row 1: Pins
        settings_grid.addWidget(QLabel("Pulse Pin:"), 1, 0)
        self.pulse_pin_spin = QSpinBox()
        self.pulse_pin_spin.setRange(2, 13)
        self.pulse_pin_spin.setValue(9)
        settings_grid.addWidget(self.pulse_pin_spin, 1, 1)

        settings_grid.addWidget(QLabel("Dir Pin:"), 1, 2)
        self.dir_pin_spin = QSpinBox()
        self.dir_pin_spin.setRange(2, 13)
        self.dir_pin_spin.setValue(8)
        settings_grid.addWidget(self.dir_pin_spin, 1, 3)

        settings_grid.addWidget(QLabel("EN Pin:"), 1, 4)
        self.en_pin_spin = QSpinBox()
        self.en_pin_spin.setRange(2, 13)
        self.en_pin_spin.setValue(7)
        settings_grid.addWidget(self.en_pin_spin, 1, 5)

        # Row 2: Motion parameters
        settings_grid.addWidget(QLabel("Rotation (°):"), 2, 0)
        self.rotation_deg_spin = QSpinBox()
        self.rotation_deg_spin.setRange(1, 3600)
        self.rotation_deg_spin.setValue(180)
        settings_grid.addWidget(self.rotation_deg_spin, 2, 1)

        settings_grid.addWidget(QLabel("Speed (steps/s):"), 2, 2)
        self.motor_speed_spin = QSpinBox()
        self.motor_speed_spin.setRange(10, 10000)
        self.motor_speed_spin.setValue(800)
        self.motor_speed_spin.setSingleStep(100)
        settings_grid.addWidget(self.motor_speed_spin, 2, 3)

        settings_grid.addWidget(QLabel("Pause (s):"), 2, 4)
        self.motor_pause_spin = QDoubleSpinBox()
        self.motor_pause_spin.setRange(0.0, 30.0)
        self.motor_pause_spin.setValue(2.0)
        self.motor_pause_spin.setSingleStep(0.5)
        settings_grid.addWidget(self.motor_pause_spin, 2, 5)

        motor_main_layout.addLayout(settings_grid)

        # Test / Stop buttons row
        motor_btn_layout = QHBoxLayout()

        self.test_motor_button = QPushButton("🔄 Test Motor")
        self.test_motor_button.setToolTip("Run one forward→reverse→pause cycle to test")
        self.test_motor_button.clicked.connect(self.test_motor_cycle)
        self.test_motor_button.setEnabled(False)
        motor_btn_layout.addWidget(self.test_motor_button)

        self.stop_motor_button = QPushButton("⛔ Stop Motor")
        self.stop_motor_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_motor_button.clicked.connect(self.stop_motor)
        self.stop_motor_button.setEnabled(False)
        motor_btn_layout.addWidget(self.stop_motor_button)

        motor_btn_layout.addStretch()
        motor_main_layout.addLayout(motor_btn_layout)

        # Motor status label
        self.motor_status_label = QLabel("Motor: Not connected")
        self.motor_status_label.setStyleSheet("color: #888; background: transparent; font-style: italic;")
        motor_main_layout.addWidget(self.motor_status_label)

        motor_group.setLayout(motor_main_layout)
        layout.addWidget(motor_group)

        # Populate serial ports
        self.refresh_serial_ports()

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

        scroll.setWidget(scroll_content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

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
            if self.batch_remaining > 0:
                self.batch_remaining -= 1

            remaining_msg = f" (Batch: {self.batch_remaining} pending)"
            self.status_message.emit(f"Recording {category}: {filename}{remaining_msg}", 0)
        else:
            self.status_message.emit(f"Recording {category}: {filename}", 0)

        self.recorder.record(device_index, duration, output_path)

    def start_next_batch_record(self):
        """Trigger the next recording in the batch."""
        if self.batch_remaining > 0 and self.batch_category:
            # If motor is enabled and connected, run motor cycle first
            if self.motor_enable_check.isChecked() and self.motor.is_connected:
                self._sync_motor_settings()
                self.motor_status_label.setText("Motor: Running cycle...")
                self.motor_status_label.setStyleSheet("color: #ff9800; background: transparent; font-style: italic;")
                self.motor.run_collection_cycle()
            else:
                self.start_recording(self.batch_category)
        else:
            self.stop_batch_recording()

    def stop_batch_recording(self):
        """Cancel any ongoing batch recording."""
        self.batch_remaining = 0
        self.batch_category = None
        self.batch_timer.stop()
        self.stop_batch_button.setVisible(False)
        
        # Stop motor if running
        if self.motor.is_connected:
            self.motor.stop()
        
        # Stop actual recording if currently recording
        if self.is_recording:
             self.recorder.stop_recording()
             
        self.status_message.emit("Batch recording stopped.", 3000)
        self.set_ui_enabled(True)

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

    # ── Motor Control Methods ──

    def refresh_serial_ports(self):
        """Scan and populate available serial ports."""
        self.serial_port_combo.clear()
        ports = MotorController.list_serial_ports()
        for port in ports:
            self.serial_port_combo.addItem(
                f"{port['device']} - {port['description']}",
                port['device']
            )
        if not ports:
            self.serial_port_combo.addItem("No ports found", None)

    def toggle_motor_connection(self):
        """Connect or disconnect the motor controller."""
        if self.motor.is_connected:
            self.motor.disconnect()
            self.connect_motor_button.setText("Connect")
            self.test_motor_button.setEnabled(False)
            self.stop_motor_button.setEnabled(False)
            self.motor_status_label.setText("Motor: Disconnected")
            self.motor_status_label.setStyleSheet("color: #888; background: transparent; font-style: italic;")
        else:
            port = self.serial_port_combo.currentData()
            if not port:
                self.status_message.emit("No serial port selected!", 3000)
                return

            # Disable button during connection to prevent double-click
            self.connect_motor_button.setEnabled(False)
            self.connect_motor_button.setText("Connecting...")
            self.motor_status_label.setText("Motor: Connecting...")
            self.motor_status_label.setStyleSheet("color: #ff9800; background: transparent; font-style: italic;")

            self._sync_motor_settings()
            baud = self.baud_rate_spin.value()

            # Run connection in background thread to avoid UI freeze
            def _connect_thread():
                success = self.motor.connect(port, baud)
                # Use signal to update UI from main thread
                if success:
                    self.motor.status_update.emit(f"__CONNECTED__{port}")
                else:
                    self.motor.status_update.emit("__CONNECT_FAILED__")

            thread = threading.Thread(target=_connect_thread, daemon=True)
            thread.start()

    def _handle_connection_result(self, message):
        """Handle connection result from background thread."""
        if message.startswith("__CONNECTED__"):
            port = message.replace("__CONNECTED__", "")
            self.connect_motor_button.setText("Disconnect")
            self.connect_motor_button.setEnabled(True)
            self.test_motor_button.setEnabled(True)
            self.stop_motor_button.setEnabled(True)
            self.motor_status_label.setText(f"Motor: Connected ({port})")
            self.motor_status_label.setStyleSheet("color: #4caf50; background: transparent; font-style: italic;")
            self.status_message.emit(f"Motor connected to {port}", 3000)
        elif message == "__CONNECT_FAILED__":
            self.connect_motor_button.setText("Connect")
            self.connect_motor_button.setEnabled(True)
            self.motor_status_label.setText("Motor: Connection failed")
            self.motor_status_label.setStyleSheet("color: #f44336; background: transparent; font-style: italic;")
        else:
            # Normal status update — show it
            self.motor_status_label.setText(f"Motor: {message}")
            self.status_message.emit(f"Motor: {message}", 3000)

    def _sync_motor_settings(self):
        """Push current UI values into the motor controller."""
        self.motor.update_config(
            pulse_pin=self.pulse_pin_spin.value(),
            dir_pin=self.dir_pin_spin.value(),
            enable_pin=self.en_pin_spin.value(),
            microsteps=self.microsteps_spin.value()
        )
        self.motor.rotation_degrees = self.rotation_deg_spin.value()
        self.motor.motor_speed = self.motor_speed_spin.value()
        self.motor.pause_after_cycle = self.motor_pause_spin.value()

    def test_motor_cycle(self):
        """Run a single forward→reverse→pause cycle to test the motor."""
        if not self.motor.is_connected:
            self.status_message.emit("Motor not connected!", 3000)
            return
        self._sync_motor_settings()
        self.test_motor_button.setEnabled(False)
        self.motor_status_label.setText("Motor: Testing cycle...")
        self.motor_status_label.setStyleSheet("color: #ff9800; background: transparent; font-style: italic;")
        self.status_message.emit("Motor test: running forward→reverse→pause cycle...", 0)
        self.motor.run_collection_cycle()

    def stop_motor(self):
        """Emergency stop the motor."""
        if self.motor.is_connected:
            self.motor.stop()
            self.test_motor_button.setEnabled(True)
            self.motor_status_label.setText("Motor: Stopped")
            self.motor_status_label.setStyleSheet("color: #f44336; background: transparent; font-style: italic;")
            self.status_message.emit("Motor stopped.", 3000)

    def on_motor_cycle_complete(self):
        """Called when motor forward→reverse→pause cycle finishes."""
        self.motor_status_label.setText("Motor: Cycle done ✔")
        self.motor_status_label.setStyleSheet("color: #4caf50; background: transparent; font-style: italic;")
        self.test_motor_button.setEnabled(True)
        # Now start the next recording (batch mode)
        if self.batch_remaining > 0 and self.batch_category:
            self.start_recording(self.batch_category)

    def on_motor_status(self, message):
        """Display motor status updates."""
        # Route connection result messages to the handler
        if message.startswith("__CONNECTED__") or message == "__CONNECT_FAILED__":
            self._handle_connection_result(message)
            return
        self.motor_status_label.setText(f"Motor: {message}")
        self.status_message.emit(f"Motor: {message}", 3000)

    def on_motor_error(self, message):
        """Handle motor errors."""
        self.motor_status_label.setText(f"Motor Error: {message}")
        self.motor_status_label.setStyleSheet("color: #f44336; background: transparent; font-style: italic;")
        self.status_message.emit(f"Motor Error: {message}", 5000)

    def cleanup(self):
        """Clean up resources."""
        self.batch_timer.stop()
        if self.motor.is_connected:
            self.motor.disconnect()
        self.recorder.cleanup()
