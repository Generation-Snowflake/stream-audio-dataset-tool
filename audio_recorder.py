#!/usr/bin/env python3
"""
Audio Dataset Recording Application
A professional desktop application for recording high-quality audio datasets.
Built with PyQt6 and PyAudio.
"""

import sys
import os
import wave
import threading
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSpinBox, QLineEdit, QProgressBar,
    QStatusBar, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QPalette, QColor, QPixmap
import pyaudio


class AudioRecorder(QObject):
    """Handles audio recording operations in a separate thread."""
    
    recording_complete = pyqtSignal(bool, str)  # success, message
    level_update = pyqtSignal(int)  # audio level (0-100)
    test_complete = pyqtSignal(bool, str)  # test recording complete
    
    def __init__(self):
        super().__init__()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.is_testing = False
        self.frames = []
        self.test_file_path = None
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000
        self.SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
        
    def get_device_list(self):
        """Get list of all available input devices."""
        devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels']
                })
        return devices
    
    def record_test(self, device_index, duration=3):
        """Record a test audio sample."""
        from pathlib import Path
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
            
            # Open stream
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            # Calculate number of chunks to record
            chunks_needed = int(self.RATE / self.CHUNK * duration)
            
            # Record
            for _ in range(chunks_needed):
                if not self.is_testing:
                    break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                
                # Update level during recording
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    mean_square = np.mean(audio_data.astype(np.float64) ** 2)
                    if mean_square >= 0:  # Ensure non-negative before sqrt
                        rms = np.sqrt(mean_square)
                        level = int((rms / 32768.0) * 100)
                        self.level_update.emit(min(100, level))
                    else:
                        self.level_update.emit(0)
                else:
                    self.level_update.emit(0)
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save to file
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
            import wave
            
            # Open the test file
            wf = wave.open(self.test_file_path, 'rb')
            
            # Open output stream
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                output=True
            )
            
            # Play audio
            data = wf.readframes(self.CHUNK)
            while data:
                stream.write(data)
                data = wf.readframes(self.CHUNK)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            wf.close()
            
        except Exception as e:
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
            
            # Open stream
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            # Calculate number of chunks to record
            chunks_needed = int(self.RATE / self.CHUNK * duration)
            
            # Record
            for _ in range(chunks_needed):
                if not self.is_recording:
                    break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                
                # Update level during recording
                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    mean_square = np.mean(audio_data.astype(np.float64) ** 2)
                    if mean_square >= 0:  # Ensure non-negative before sqrt
                        rms = np.sqrt(mean_square)
                        level = int((rms / 32768.0) * 100)
                        self.level_update.emit(min(100, level))
                    else:
                        self.level_update.emit(0)
                else:
                    self.level_update.emit(0)
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save to file
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


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.current_index = 1
        self.is_testing = False
        self.is_recording = False
        
        
        self.init_ui()
        self.load_devices()
        
        # Connect signals
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.level_update.connect(self.update_level_meter)
        self.recorder.test_complete.connect(self.on_test_complete)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Audio Dataset Recorder")
        self.setGeometry(100, 100, 600, 500)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Logo at the top center
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo to a reasonable size (e.g., 120x120)
            scaled_pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_label.setStyleSheet("background: transparent; padding: 10px;")
            main_layout.addWidget(logo_label)
        
        # Device Selection Group
        device_group = QGroupBox("Audio Device")
        device_layout = QVBoxLayout()
        
        self.device_combo = QComboBox()
        device_layout.addWidget(QLabel("Microphone:"))
        device_layout.addWidget(self.device_combo)
        
        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)
        
        # Level Meter Group
        meter_group = QGroupBox("Audio Test")
        meter_layout = QVBoxLayout()
        
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(True)
        self.level_meter.setFormat("%v%")
        
        # Test buttons layout
        test_buttons_layout = QHBoxLayout()
        
        self.test_record_button = QPushButton("Record Test (3s)")
        self.test_record_button.clicked.connect(self.record_test)
        
        self.test_play_button = QPushButton("Play Test")
        self.test_play_button.clicked.connect(self.play_test)
        self.test_play_button.setEnabled(False)  # Disabled until test is recorded
        
        test_buttons_layout.addWidget(self.test_record_button)
        test_buttons_layout.addWidget(self.test_play_button)
        
        meter_layout.addWidget(self.level_meter)
        meter_layout.addLayout(test_buttons_layout)
        
        meter_group.setLayout(meter_layout)
        main_layout.addWidget(meter_group)
        
        # Configuration Group
        config_group = QGroupBox("Recording Configuration")
        config_layout = QVBoxLayout()
        
        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (seconds):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)
        self.duration_spin.setValue(5)
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
        
        # Next filename display
        self.next_filename_label = QLabel()
        self.update_next_filename_display()
        config_layout.addWidget(self.next_filename_label)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
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
        
        main_layout.addLayout(buttons_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        main_layout.addStretch()
    
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        dark_stylesheet = """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }
            QGroupBox {
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                font-weight: bold;
                color: #00bcd4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
            QLabel {
                color: #b0b0b0;
                background: transparent;
            }
            QPushButton {
                background-color: #00bcd4;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #00acc1;
            }
            QPushButton:pressed {
                background-color: #0097a7;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #6a6a6a;
            }
            QComboBox, QSpinBox, QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
            }
            QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
                border: 1px solid #00bcd4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #4caf50, stop:0.5 #ffeb3b, stop:1 #f44336);
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #252525;
                color: #b0b0b0;
            }
        """
        self.setStyleSheet(dark_stylesheet)
    

    
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
            self.status_bar.showMessage("No input devices found!", 5000)
    
    def record_test(self):
        """Record a test audio sample."""
        if self.is_testing or self.is_recording:
            return
        
        device_index = self.device_combo.currentData()
        if device_index is None:
            self.status_bar.showMessage("No device selected!", 3000)
            return
        
        # Disable buttons during test
        self.is_testing = True
        self.test_record_button.setEnabled(False)
        self.test_play_button.setEnabled(False)
        self.device_combo.setEnabled(False)
        
        # Update button text
        self.test_record_button.setText("Recording... (3s)")
        self.status_bar.showMessage("Recording test sample...")
        
        # Start recording
        self.recorder.record_test(device_index, duration=3)
    
    def play_test(self):
        """Play back the test recording."""
        if self.is_testing or self.is_recording:
            return
        
        if self.recorder.play_test():
            self.test_play_button.setEnabled(False)
            self.status_bar.showMessage("Playing test recording...")
            
            # Re-enable after 3 seconds (duration of test)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(3000, lambda: self.test_play_button.setEnabled(True))
            QTimer.singleShot(3000, lambda: self.status_bar.showMessage("Playback complete", 3000))
        else:
            self.status_bar.showMessage("No test recording available!", 3000)
    
    def on_test_complete(self, success, message):
        """Handle test recording completion."""
        self.is_testing = False
        self.test_record_button.setEnabled(True)
        self.test_record_button.setText("Record Test (3s)")
        self.device_combo.setEnabled(True)
        self.level_meter.setValue(0)
        
        if success:
            self.test_play_button.setEnabled(True)
            self.status_bar.showMessage("Test recorded! Click 'Play Test' to listen.", 5000)
        else:
            self.status_bar.showMessage(message, 5000)
    
    def update_level_meter(self, level):
        """Update level meter (called from recording thread)."""
        self.level_meter.setValue(level)
    
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
        
        # Stop testing if active (no action needed - test recording completes automatically)
        
        # Get parameters
        device_index = self.device_combo.currentData()
        if device_index is None:
            self.status_bar.showMessage("No device selected!", 3000)
            return
        
        duration = self.duration_spin.value()
        prefix = self.prefix_input.text() or "sample"
        
        # Create output directory
        output_dir = Path("output") / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{prefix}_{self.current_index}.wav"
        output_path = str(output_dir / filename)
        
        # Update UI
        self.is_recording = True
        self.set_ui_enabled(False)
        
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
        
        self.status_bar.showMessage(f"Recording {category}: {filename}")
        
        # Start recording
        self.recorder.record(device_index, duration, output_path)
    
    def on_recording_complete(self, success, message):
        """Handle recording completion."""
        self.is_recording = False
        self.set_ui_enabled(True)
        
        # Reset button styles
        self.record_ok_button.setText("Record OK")
        self.record_ng_button.setText("Record NG")
        self.record_ok_button.setStyleSheet("")
        self.record_ng_button.setStyleSheet("")
        
        # Update status
        if success:
            self.status_bar.showMessage(message, 5000)
            
            # Auto-increment index
            self.current_index += 1
            self.index_spin.setValue(self.current_index)
        else:
            self.status_bar.showMessage(f"Recording failed: {message}", 5000)
        
        # Reset level meter
        self.level_meter.setValue(0)
    
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements."""
        self.device_combo.setEnabled(enabled)
        self.test_record_button.setEnabled(enabled)
        self.duration_spin.setEnabled(enabled)
        self.prefix_input.setEnabled(enabled)
        self.index_spin.setEnabled(enabled)
        self.record_ok_button.setEnabled(enabled)
        self.record_ng_button.setEnabled(enabled)
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.recorder.cleanup()
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Dataset Recorder")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
