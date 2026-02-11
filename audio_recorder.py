#!/usr/bin/env python3
"""
Audio Dataset Tool
A professional desktop application for recording, analyzing, training,
and testing audio datasets for sound classification.

Built with PyQt6 and integrates YAMNet-based ML capabilities.
"""

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QStatusBar, QWidget, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from ui.recorder_tab import RecorderWidget
from ui.analysis_tab import AnalysisWidget
from ui.training_tab import TrainingWidget
from ui.inference_tab import InferenceWidget


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Audio Dataset Tool")
        self.setGeometry(100, 100, 1000, 750)

        # Apply dark theme
        self.apply_dark_theme()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            scaled_pixmap = pixmap.scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_label.setStyleSheet("background: transparent; padding: 5px;")
            main_layout.addWidget(logo_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #b0b0b0;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #00bcd4;
                border-bottom: 2px solid #00bcd4;
            }
            QTabBar::tab:hover {
                background-color: #353535;
                color: #e0e0e0;
            }
        """)

        # Create tabs
        self.recorder_tab = RecorderWidget()
        self.analysis_tab = AnalysisWidget()
        self.training_tab = TrainingWidget()
        self.inference_tab = InferenceWidget()

        self.tab_widget.addTab(self.recorder_tab, "🎙️ Recording")
        self.tab_widget.addTab(self.analysis_tab, "📊 Analysis")
        self.tab_widget.addTab(self.training_tab, "🧠 Training")
        self.tab_widget.addTab(self.inference_tab, "🔍 Inference")

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Connect status signals from all tabs
        self.recorder_tab.status_message.connect(self.show_status)
        self.analysis_tab.status_message.connect(self.show_status)
        self.training_tab.status_message.connect(self.show_status)
        self.inference_tab.status_message.connect(self.show_status)

    def show_status(self, message, timeout):
        """Show status bar message."""
        self.status_bar.showMessage(message, timeout)

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
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 8px;
                color: #e0e0e0;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
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
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                color: #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #00bcd4;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #353535;
            }
            QSplitter::handle {
                background-color: #3a3a3a;
                width: 3px;
            }
        """
        self.setStyleSheet(dark_stylesheet)

    def closeEvent(self, event):
        """Handle window close event."""
        self.recorder_tab.cleanup()
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Dataset Tool")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
