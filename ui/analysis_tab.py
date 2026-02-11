"""
Analysis Tab Widget - Audio file analysis with energy comparison.
Analyzes both OK and NG folders together for comparison.
Integrates functionality from sound-analysis project.
"""

import os
import threading
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QFileDialog,
    QListWidget, QListWidgetItem,
    QSplitter, QTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class AnalysisWidget(QWidget):
    """Analysis tab widget for analyzing OK and NG audio datasets together."""

    status_message = pyqtSignal(str, int)
    _update_metrics_signal = pyqtSignal(dict)
    _batch_log_signal = pyqtSignal(str)
    _batch_progress_signal = pyqtSignal(int)
    _batch_done_signal = pyqtSignal()
    _update_energy_graph_signal = pyqtSignal(list, list)  # ok_results, ng_results

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ok_files = []
        self.ng_files = []
        self.current_files = []  # combined list with category tags
        self.current_analysis = None
        self.init_ui()

        # Connect thread-safe signals
        self._update_metrics_signal.connect(self._do_update_metrics)
        self._batch_log_signal.connect(self._do_batch_log)
        self._batch_progress_signal.connect(self._do_batch_progress)
        self._batch_done_signal.connect(self._do_batch_done)
        self._update_energy_graph_signal.connect(self._do_update_energy_graph)

    def init_ui(self):
        """Initialize the analysis UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Top row: folder selection (select output/ parent)
        folder_group = QGroupBox("Dataset Folder (contains OK/ and NG/)")
        folder_layout = QHBoxLayout()

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: #b0b0b0; background: transparent;")
        folder_layout.addWidget(self.folder_label, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(120)
        browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_btn)

        load_default_btn = QPushButton("Load output/")
        load_default_btn.setMaximumWidth(120)
        load_default_btn.clicked.connect(self.load_default_folder)
        folder_layout.addWidget(load_default_btn)

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Dataset info bar
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #a6adc8; background: transparent; padding: 2px 5px;")
        layout.addWidget(self.info_label)

        # Splitter: file list on left, visualization on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: file list + metrics
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        file_group = QGroupBox("Audio Files (OK + NG)")
        file_layout = QVBoxLayout()

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)

        self.analyze_all_btn = QPushButton("📊 Analyze All (OK + NG)")
        self.analyze_all_btn.clicked.connect(self.analyze_all_files)
        self.analyze_all_btn.setEnabled(False)
        file_layout.addWidget(self.analyze_all_btn)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Metrics display
        metrics_group = QGroupBox("Selected File Metrics")
        metrics_layout = QVBoxLayout()

        self.file_info_label = QLabel("File: --")
        self.file_info_label.setStyleSheet("color: #cdd6f4; font-weight: bold; background: transparent;")
        metrics_layout.addWidget(self.file_info_label)

        self.category_label = QLabel("Category: --")
        self.category_label.setStyleSheet("color: #74c7ec; background: transparent;")
        metrics_layout.addWidget(self.category_label)

        self.energy_label = QLabel("Average Energy: --")
        self.energy_label.setStyleSheet("color: #74c7ec; background: transparent;")
        metrics_layout.addWidget(self.energy_label)

        self.variance_label = QLabel("Temporal Variance: --")
        self.variance_label.setStyleSheet("color: #74c7ec; background: transparent;")
        metrics_layout.addWidget(self.variance_label)

        self.rms_label = QLabel("RMS: --")
        self.rms_label.setStyleSheet("color: #74c7ec; background: transparent;")
        metrics_layout.addWidget(self.rms_label)

        self.duration_label = QLabel("Duration: --")
        self.duration_label.setStyleSheet("color: #74c7ec; background: transparent;")
        metrics_layout.addWidget(self.duration_label)

        metrics_group.setLayout(metrics_layout)
        left_layout.addWidget(metrics_group)

        splitter.addWidget(left_widget)

        # Right: visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Energy comparison graph
        energy_group = QGroupBox("Energy Comparison Graph (OK vs NG)")
        energy_layout = QVBoxLayout()

        self.energy_figure = Figure(figsize=(8, 5), facecolor='#1e1e1e')
        self.energy_ax = self.energy_figure.add_subplot(111)
        self.energy_ax.set_facecolor('#1e1e1e')
        self.energy_ax.set_title('Run batch analysis to see energy graph', color='#a6adc8', fontsize=11)
        self.energy_ax.tick_params(colors='#a6adc8')
        self.energy_canvas = FigureCanvas(self.energy_figure)
        energy_layout.addWidget(self.energy_canvas)

        energy_group.setLayout(energy_layout)
        right_layout.addWidget(energy_group)

        # Batch results
        batch_group = QGroupBox("Batch Analysis Results (OK vs NG Comparison)")
        batch_layout = QVBoxLayout()

        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress)

        self.batch_results = QTextEdit()
        self.batch_results.setReadOnly(True)
        self.batch_results.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px;
                font-family: monospace; font-size: 10pt;
            }
        """)
        batch_layout.addWidget(self.batch_results)

        batch_group.setLayout(batch_layout)
        right_layout.addWidget(batch_group)

        splitter.addWidget(right_widget)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def browse_folder(self):
        """Browse for the dataset root folder (parent of OK/ and NG/)."""
        default_dir = os.path.join(os.getcwd(), "output")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Root (containing OK/ and NG/)", default_dir
        )
        if folder:
            self.load_dataset_folder(folder)

    def load_default_folder(self):
        """Load the default output/ folder."""
        default_dir = os.path.join(os.getcwd(), "output")
        if os.path.exists(default_dir):
            self.load_dataset_folder(default_dir)
        else:
            self.status_message.emit("output/ folder not found", 3000)

    def load_dataset_folder(self, folder_path):
        """Load audio files from both OK/ and NG/ subdirectories."""
        self.folder_label.setText(folder_path)
        self.file_list.clear()
        self.ok_files = []
        self.ng_files = []
        self.current_files = []

        ok_dir = os.path.join(folder_path, "OK")
        ng_dir = os.path.join(folder_path, "NG")

        # Load OK files
        if os.path.exists(ok_dir):
            ok_wavs = sorted([f for f in os.listdir(ok_dir) if f.lower().endswith('.wav')])
            for f in ok_wavs:
                full_path = os.path.join(ok_dir, f)
                self.ok_files.append(full_path)
                self.current_files.append(('OK', full_path))
        else:
            ok_wavs = []

        # Load NG files
        if os.path.exists(ng_dir):
            ng_wavs = sorted([f for f in os.listdir(ng_dir) if f.lower().endswith('.wav')])
            for f in ng_wavs:
                full_path = os.path.join(ng_dir, f)
                self.ng_files.append(full_path)
                self.current_files.append(('NG', full_path))
        else:
            ng_wavs = []

        # Populate list with color-coded items
        for category, file_path in self.current_files:
            filename = os.path.basename(file_path)
            item = QListWidgetItem(f"[{category}] {filename}")

            if category == 'OK':
                item.setForeground(QColor("#a6e3a1"))  # Green
            else:
                item.setForeground(QColor("#f38ba8"))  # Red

            self.file_list.addItem(item)

        total = len(self.current_files)
        info_text = f"OK: {len(self.ok_files)} files  |  NG: {len(self.ng_files)} files  |  Total: {total} files"
        self.info_label.setText(info_text)

        if total > 0:
            self.analyze_all_btn.setEnabled(True)
            self.status_message.emit(f"Loaded {total} files ({len(self.ok_files)} OK, {len(self.ng_files)} NG)", 3000)
        else:
            self.analyze_all_btn.setEnabled(False)
            self.status_message.emit("No audio files found in OK/ or NG/ subfolders", 3000)

    def on_file_selected(self, index):
        """Handle file selection from the list."""
        if index < 0 or index >= len(self.current_files):
            return

        category, file_path = self.current_files[index]
        self.analyze_single_file(file_path, category)

    def analyze_single_file(self, file_path, category=""):
        """Analyze a single audio file and update UI."""
        try:
            from utils.audio_analysis import analyze_audio_file

            result = analyze_audio_file(file_path)
            result['category'] = category
            result['filename'] = os.path.basename(file_path)
            self.current_analysis = result

            # Update metrics (thread-safe via signal)
            self._update_metrics_signal.emit(result)

            self.status_message.emit(f"Analyzed: [{category}] {os.path.basename(file_path)}", 3000)

        except Exception as e:
            self.status_message.emit(f"Error analyzing file: {str(e)}", 5000)

    def _do_update_metrics(self, result):
        """Update metrics labels (must run on main thread)."""
        category = result.get('category', '')
        filename = result.get('filename', '')

        self.file_info_label.setText(f"File: {filename}")

        if category == 'OK':
            self.category_label.setText(f"Category: ✅ OK")
            self.category_label.setStyleSheet("color: #a6e3a1; font-weight: bold; background: transparent;")
        elif category == 'NG':
            self.category_label.setText(f"Category: ❌ NG")
            self.category_label.setStyleSheet("color: #f38ba8; font-weight: bold; background: transparent;")
        else:
            self.category_label.setText(f"Category: --")
            self.category_label.setStyleSheet("color: #74c7ec; background: transparent;")

        self.energy_label.setText(f"Average Energy: {result['energy']:.2f} dB")
        self.variance_label.setText(f"Temporal Variance: {result['variance']:.2f}")
        self.rms_label.setText(f"RMS: {result['rms']:.4f}")
        self.duration_label.setText(f"Duration: {result['duration']:.2f}s")

    def analyze_all_files(self):
        """Analyze all OK and NG files together with comparison."""
        if not self.current_files:
            return

        self.analyze_all_btn.setEnabled(False)
        self.batch_progress.setVisible(True)
        self.batch_results.clear()

        thread = threading.Thread(target=self._analyze_all_thread, daemon=True)
        thread.start()

    def _do_batch_log(self, text):
        """Append to batch results (thread-safe)."""
        self.batch_results.append(text)
        scrollbar = self.batch_results.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _do_batch_progress(self, value):
        """Update batch progress (thread-safe)."""
        self.batch_progress.setValue(value)

    def _do_batch_done(self):
        """Handle batch completion (thread-safe)."""
        self.batch_progress.setVisible(False)
        self.analyze_all_btn.setEnabled(True)

    def _do_update_energy_graph(self, ok_results, ng_results):
        """Draw scatter plot (point cloud) comparing OK vs NG (must run on main thread)."""
        try:
            from matplotlib.patches import Ellipse

            self.energy_ax.clear()
            self.energy_ax.set_facecolor('#1e1e1e')

            ok_data = [(r['file'], r['energy'], r['variance']) for r in ok_results if 'energy' in r]
            ng_data = [(r['file'], r['energy'], r['variance']) for r in ng_results if 'energy' in r]

            if not ok_data and not ng_data:
                self.energy_ax.set_title('No data', color='#a6adc8', fontsize=11)
                self.energy_canvas.draw()
                return

            # Plot OK points
            if ok_data:
                ok_names, ok_energies, ok_variances = zip(*ok_data)
                ok_energies = np.array(ok_energies)
                ok_variances = np.array(ok_variances)
                self.energy_ax.scatter(
                    ok_energies, ok_variances,
                    c='#a6e3a1', s=80, alpha=0.85, edgecolors='white',
                    linewidths=0.5, label=f'OK ({len(ok_data)})', zorder=5
                )
                # Label each point
                for name, ex, vx in ok_data:
                    short = name.replace('.wav', '')
                    self.energy_ax.annotate(
                        short, (ex, vx), fontsize=6, color='#a6e3a1',
                        alpha=0.7, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points'
                    )
                # Draw cluster ellipse
                if len(ok_data) >= 2:
                    ok_mean_e, ok_mean_v = np.mean(ok_energies), np.mean(ok_variances)
                    ok_std_e, ok_std_v = np.std(ok_energies), np.std(ok_variances)
                    ellipse = Ellipse(
                        (ok_mean_e, ok_mean_v),
                        width=max(ok_std_e * 4, 0.5), height=max(ok_std_v * 4, 0.01),
                        fill=False, edgecolor='#a6e3a1', linestyle='--',
                        linewidth=1.5, alpha=0.5
                    )
                    self.energy_ax.add_patch(ellipse)
                    # Mean marker
                    self.energy_ax.scatter(
                        [ok_mean_e], [ok_mean_v],
                        c='#a6e3a1', s=200, marker='X', edgecolors='white',
                        linewidths=1.5, zorder=6, alpha=0.9
                    )

            # Plot NG points
            if ng_data:
                ng_names, ng_energies, ng_variances = zip(*ng_data)
                ng_energies = np.array(ng_energies)
                ng_variances = np.array(ng_variances)
                self.energy_ax.scatter(
                    ng_energies, ng_variances,
                    c='#f38ba8', s=80, alpha=0.85, edgecolors='white',
                    linewidths=0.5, label=f'NG ({len(ng_data)})', zorder=5
                )
                # Label each point
                for name, ex, vx in ng_data:
                    short = name.replace('.wav', '')
                    self.energy_ax.annotate(
                        short, (ex, vx), fontsize=6, color='#f38ba8',
                        alpha=0.7, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points'
                    )
                # Draw cluster ellipse
                if len(ng_data) >= 2:
                    ng_mean_e, ng_mean_v = np.mean(ng_energies), np.mean(ng_variances)
                    ng_std_e, ng_std_v = np.std(ng_energies), np.std(ng_variances)
                    ellipse = Ellipse(
                        (ng_mean_e, ng_mean_v),
                        width=max(ng_std_e * 4, 0.5), height=max(ng_std_v * 4, 0.01),
                        fill=False, edgecolor='#f38ba8', linestyle='--',
                        linewidth=1.5, alpha=0.5
                    )
                    self.energy_ax.add_patch(ellipse)
                    # Mean marker
                    self.energy_ax.scatter(
                        [ng_mean_e], [ng_mean_v],
                        c='#f38ba8', s=200, marker='X', edgecolors='white',
                        linewidths=1.5, zorder=6, alpha=0.9
                    )

            # Styling
            self.energy_ax.set_xlabel('Energy (dB)', color='#a6adc8', fontsize=10)
            self.energy_ax.set_ylabel('Temporal Variance', color='#a6adc8', fontsize=10)
            self.energy_ax.set_title('OK vs NG — Energy × Variance', color='#cdd6f4', fontsize=12, pad=10)
            self.energy_ax.tick_params(colors='#a6adc8')
            self.energy_ax.legend(loc='upper right', fontsize=9, facecolor='#2d2d2d',
                                  edgecolor='#3a3a3a', labelcolor='#a6adc8')
            self.energy_ax.spines['top'].set_color('#3a3a3a')
            self.energy_ax.spines['right'].set_color('#3a3a3a')
            self.energy_ax.spines['bottom'].set_color('#3a3a3a')
            self.energy_ax.spines['left'].set_color('#3a3a3a')
            self.energy_ax.grid(True, alpha=0.15, color='#585b70')

            self.energy_figure.tight_layout()
            self.energy_canvas.draw()

        except Exception as e:
            print(f"Energy graph error: {e}")

    def _analyze_all_thread(self):
        """Thread function for batch analysis of both OK and NG."""
        from utils.audio_analysis import analyze_audio_file

        ok_results = []
        ng_results = []
        total = len(self.current_files)

        self._batch_log_signal.emit("=" * 50)
        self._batch_log_signal.emit("📊 Batch Analysis - OK vs NG Comparison")
        self._batch_log_signal.emit("=" * 50)

        for i, (category, file_path) in enumerate(self.current_files):
            try:
                analysis = analyze_audio_file(file_path)

                entry = {
                    'file': os.path.basename(file_path),
                    'category': category,
                    'energy': analysis['energy'],
                    'variance': analysis['variance'],
                    'rms': analysis['rms'],
                }

                if category == 'OK':
                    ok_results.append(entry)
                else:
                    ng_results.append(entry)

            except Exception as e:
                entry = {
                    'file': os.path.basename(file_path),
                    'category': category,
                    'error': str(e)
                }
                if category == 'OK':
                    ok_results.append(entry)
                else:
                    ng_results.append(entry)

            self._batch_progress_signal.emit(int((i + 1) / total * 100))

        # === Format OK Results ===
        self._batch_log_signal.emit(f"\n{'─'*50}")
        self._batch_log_signal.emit(f"✅ OK Samples ({len(ok_results)} files)")
        self._batch_log_signal.emit(f"{'─'*50}")

        ok_energies = []
        ok_variances = []
        for r in ok_results:
            if 'error' in r:
                self._batch_log_signal.emit(f"  ❌ {r['file']}: ERROR - {r['error']}")
            else:
                ok_energies.append(r['energy'])
                ok_variances.append(r['variance'])
                self._batch_log_signal.emit(
                    f"  {r['file']}: Energy={r['energy']:.2f} dB | Var={r['variance']:.2f} | RMS={r['rms']:.4f}"
                )

        # === Format NG Results ===
        self._batch_log_signal.emit(f"\n{'─'*50}")
        self._batch_log_signal.emit(f"❌ NG Samples ({len(ng_results)} files)")
        self._batch_log_signal.emit(f"{'─'*50}")

        ng_energies = []
        ng_variances = []
        for r in ng_results:
            if 'error' in r:
                self._batch_log_signal.emit(f"  ❌ {r['file']}: ERROR - {r['error']}")
            else:
                ng_energies.append(r['energy'])
                ng_variances.append(r['variance'])
                self._batch_log_signal.emit(
                    f"  {r['file']}: Energy={r['energy']:.2f} dB | Var={r['variance']:.2f} | RMS={r['rms']:.4f}"
                )

        # === Comparison Summary ===
        self._batch_log_signal.emit(f"\n{'='*50}")
        self._batch_log_signal.emit("📈 COMPARISON SUMMARY")
        self._batch_log_signal.emit(f"{'='*50}")

        if ok_energies:
            ok_avg_energy = np.mean(ok_energies)
            ok_avg_var = np.mean(ok_variances)
            self._batch_log_signal.emit(f"  OK  → Avg Energy: {ok_avg_energy:.2f} dB | Avg Variance: {ok_avg_var:.2f}")
        else:
            self._batch_log_signal.emit(f"  OK  → No valid samples")

        if ng_energies:
            ng_avg_energy = np.mean(ng_energies)
            ng_avg_var = np.mean(ng_variances)
            self._batch_log_signal.emit(f"  NG  → Avg Energy: {ng_avg_energy:.2f} dB | Avg Variance: {ng_avg_var:.2f}")
        else:
            self._batch_log_signal.emit(f"  NG  → No valid samples")

        if ok_energies and ng_energies:
            energy_gap = ng_avg_energy - ok_avg_energy
            self._batch_log_signal.emit(f"\n  Energy Gap (NG - OK): {energy_gap:+.2f} dB")

        self._batch_log_signal.emit(f"\n{'='*50}")

        # Emit energy graph data
        self._update_energy_graph_signal.emit(ok_results, ng_results)

        self.status_message.emit(
            f"Analyzed {total} files: {len(ok_results)} OK, {len(ng_results)} NG", 5000
        )
        self._batch_done_signal.emit()
