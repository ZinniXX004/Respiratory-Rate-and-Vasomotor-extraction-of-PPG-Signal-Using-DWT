"""
-+-+-+ HRV ANALYSIS GUI APPLICATION +-+-+-
This PyQt6 GUI provides a comprehensive interface for PPG signal analysis,
importing its core logic from 'PPG_main_logic.py'.

--- MODIFICATION HIGHLIGHTS ---
- [DWT PLOT FIX] Changed the y-axis label for DWT Frequency Response plots
  from "Magnitude (dB)" to "Magnitude" and removed the fixed y-axis limits
  to correctly display the linear magnitude from the new backend logic.
- [RESTORED] Re-added the Respiratory Rate and Vasomotor Activity estimations
  to the results panel.
- [DWT REWORK] The AnalysisThread now calls the new, slide-accurate frequency
  response generation method from the backend.
"""

import sys
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit,
                             QSpinBox, QFileDialog, QTextEdit, QGroupBox, QGridLayout,
                             QStatusBar, QProgressBar, QComboBox, QCheckBox, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PPG_main_fixed import (PPGStressAnalyzer, HRV_Analyzer, welch_from_scratch, 
                              extract_rate_from_signal)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class MplCanvasWithToolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.toolbar.setStyleSheet("background-color: #4a5568;")

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2d3748')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.apply_dark_theme()

    def apply_dark_theme(self):
        self.axes.set_facecolor('#1a202c')
        for spine in self.axes.spines.values(): spine.set_color('#a0aec0')
        self.axes.xaxis.label.set_color('#a0aec0')
        self.axes.yaxis.label.set_color('#a0aec0')
        self.axes.title.set_color('white')
        self.axes.tick_params(axis='x', colors='#a0aec0')
        self.axes.tick_params(axis='y', colors='#a0aec0')
        self.fig.tight_layout()

    def plot_data(self, plot_instructions):
        self.axes.clear()
        for instruction in plot_instructions:
            plot_type = instruction.get('type', 'plot')
            style = instruction.get('style', {})
            x_data, y_data = instruction.get('x', []), instruction.get('y', [])
            if plot_type == 'axhline': self.axes.axhline(y=instruction.get('y', 0), label=instruction.get('label'), **style)
            elif plot_type == 'axvline': self.axes.axvline(x=instruction.get('x', 0), label=instruction.get('label'), **style)
            elif plot_type == 'fill_between': self.axes.fill_between(x_data, instruction.get('y1'), instruction.get('y2', 0), label=instruction.get('label'), **style)
            elif len(x_data) > 0 and len(y_data) > 0:
                if plot_type == 'scatter': self.axes.scatter(x_data, y_data, label=instruction.get('label'), **style)
                elif plot_type == 'bar': self.axes.bar(x_data, y_data, label=instruction.get('label'), **style)
                else: self.axes.plot(x_data, y_data, label=instruction.get('label'), **style)
        self.axes.set_title(plot_instructions[0].get('title', ''))
        self.axes.set_xlabel(plot_instructions[0].get('xlabel', ''))
        self.axes.set_ylabel(plot_instructions[0].get('ylabel', ''))
        if xlim := plot_instructions[0].get('xlim'): self.axes.set_xlim(xlim)
        if ylim := plot_instructions[0].get('ylim'): self.axes.set_ylim(ylim)
        if any('label' in p for p in plot_instructions):
            legend = self.axes.legend()
            if legend:
                legend.get_frame().set_facecolor('#2d3748')
                plt.setp(legend.get_texts(), color='#a0aec0')
        self.axes.grid(True, linestyle='--', alpha=0.2)
        self.apply_dark_theme()
        self.draw()

class AnalysisThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, analyzer_instance, file_path, column_name, downsample, ds_factor, hrv_source_str, rr_source_str, vma_source_str):
        super().__init__()
        self.analyzer, self.file_path, self.column_name = analyzer_instance, file_path, column_name
        self.downsample, self.ds_factor = downsample, ds_factor if downsample else 1
        self.hrv_source_str, self.rr_source_str, self.vma_source_str = hrv_source_str, rr_source_str, vma_source_str

    def _get_signal_from_source(self, source_str, base_signal, dwt_coeffs):
        if source_str == 'Journal Pipeline': return base_signal
        try:
            level = int(source_str.split(' ')[-1].replace('Q', ''))
            return dwt_coeffs.get(level)
        except (ValueError, IndexError): return None

    def run(self):
        try:
            self.progress.emit(5, f"Loading '{self.column_name}'...")
            time, signal = self.analyzer.load_ppg_data(self.file_path, self.column_name)
            if signal is None: self.error.emit(f"Could not read '{self.column_name}' column."); return
            
            original_fs_temp = self.analyzer.original_fs
            self.analyzer.fs = original_fs_temp
            self.progress.emit(10, "Calculating original signal FFT...")
            fft_original = self.analyzer.fft_magnitude_and_frequencies(signal)

            self.progress.emit(15, "Downsampling signal...")
            ds_signal, ds_time = self.analyzer.downsample_signal(signal, time, self.ds_factor)
            
            self.progress.emit(20, "Calculating downsampled signal FFT...")
            fft_downsampled = self.analyzer.fft_magnitude_and_frequencies(ds_signal)
            
            self.progress.emit(30, "Preprocessing and DWT...")
            temp_hrv_analyzer = HRV_Analyzer(ds_signal, ds_time, self.analyzer.fs, self.analyzer.fft_from_scratch)
            temp_hrv_analyzer._preprocess_and_filter()
            dwt_coeffs = self.analyzer.dwt_convolution_from_scratch(temp_hrv_analyzer.preprocessed_signal)

            self.progress.emit(45, f"Running HRV pipeline on '{self.hrv_source_str}'...")
            hrv_input_signal = self._get_signal_from_source(self.hrv_source_str, ds_signal, dwt_coeffs)
            if hrv_input_signal is None: self.error.emit(f"HRV source '{self.hrv_source_str}' not available."); return
            hrv_analyzer = HRV_Analyzer(hrv_input_signal, ds_time, self.analyzer.fs, self.analyzer.fft_from_scratch)
            hrv_results = hrv_analyzer.run_all_analyses()
            
            self.progress.emit(60, "Calculating pre-processed FFT...")
            fft_processed = self.analyzer.fft_magnitude_and_frequencies(hrv_analyzer.preprocessed_signal)

            self.progress.emit(70, f"Extracting Respiratory Rate from '{self.rr_source_str}'...")
            rr_input_signal = self._get_signal_from_source(self.rr_source_str, ds_signal, dwt_coeffs)
            respiratory_rate_brpm = extract_rate_from_signal(rr_input_signal, self.analyzer.fs, (0.15, 0.4), self.analyzer.fft_from_scratch)

            self.progress.emit(80, f"Extracting Vasomotor Activity from '{self.vma_source_str}'...")
            vma_input_signal = self._get_signal_from_source(self.vma_source_str, ds_signal, dwt_coeffs)
            vasomotor_rate_cpm = extract_rate_from_signal(vma_input_signal, self.analyzer.fs, (0.04, 0.15), self.analyzer.fft_from_scratch)
            vma_psd = welch_from_scratch(vma_input_signal, self.analyzer.fs, fft_func=self.analyzer.fft_from_scratch)

            self.progress.emit(90, "Calculating DWT filter responses...")
            orig_responses = self.analyzer.calculate_qj_frequency_responses(original_fs_temp)
            ds_responses = self.analyzer.calculate_qj_frequency_responses(self.analyzer.fs)
            dwt_freq_responses = {i: {'orig': orig_responses.get(i), 'ds': ds_responses.get(i)} for i in range(1, 9)}
            
            self.progress.emit(100, "Finalizing...")
            final_results = {'raw_signal': signal, 'raw_time': time, 'fft_original': fft_original, 'ds_signal': ds_signal, 'ds_time': ds_time, 'fft_downsampled': fft_downsampled, 'preprocessed_signal_main': hrv_analyzer.preprocessed_signal, 'preprocessed_time': ds_time, 'fft_processed': fft_processed, 'hrv_results': hrv_results, 'dwt_results': dwt_coeffs, 'dwt_freq_responses': dwt_freq_responses, 'vasomotor_psd': vma_psd, 'original_fs': original_fs_temp, 'downsampled_fs': self.analyzer.fs, 'respiratory_rate_brpm': respiratory_rate_brpm, 'vasomotor_rate_cpm': vasomotor_rate_cpm, 'hrv_source_str': self.hrv_source_str}
            self.finished.emit(final_results)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = PPGStressAnalyzer()
        self.results = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PPG HRV & DWT Analysis Dashboard - Jeremia Manalu (5023231017)")
        self.setGeometry(50, 50, 1800, 1000)
        self.setStyleSheet("""QWidget { background-color: #2d3748; color: #e2e8f0; font-size: 14px; } QMainWindow { background-color: #1a202c; } QTabWidget::pane { border: none; } QTabBar::tab { background-color: #4a5568; padding: 12px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; font-weight: bold; } QTabBar::tab:selected { background-color: #2d3748; border-bottom: 3px solid #63b3ed; } QGroupBox { border: 1px solid #4a5568; border-radius: 8px; margin-top: 1ex; font-weight: bold; font-size: 16px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; } QPushButton { background-color: #2b6cb0; border-radius: 5px; padding: 8px; font-weight: bold; border: 1px solid #2c5282; } QPushButton:hover { background-color: #3182ce; } QPushButton:disabled { background-color: #4a5568; color: #718096; } QLabel { margin-top: 5px; } QLineEdit, QSpinBox, QComboBox { background-color: #1a202c; border: 1px solid #4a5568; padding: 5px; border-radius: 5px; } QTextEdit { background-color: #1a202c; border: 1px solid #4a5568; font-family: "Consolas", "Courier New", monospace; } QProgressBar { border-radius: 5px; text-align: center; color: black; font-weight: bold; } QProgressBar::chunk { background-color: #68d391; border-radius: 5px;} QStatusBar { font-weight: bold; } QScrollArea { border: none; }""")
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        top_layout = QHBoxLayout(); top_layout.addWidget(self._create_control_panel()); self.tabs = QTabWidget(); self._create_tabs(); top_layout.addWidget(self.tabs); main_layout.addLayout(top_layout)
        bottom_bar = QHBoxLayout(); self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.progress_bar = QProgressBar(); self.status_bar.addPermanentWidget(self.progress_bar, 1); self.progress_bar.hide()
        clear_btn = QPushButton("Clear All"); exit_btn = QPushButton("Exit Application"); clear_btn.setFixedWidth(120); exit_btn.setFixedWidth(150)
        bottom_bar.addStretch(); bottom_bar.addWidget(clear_btn); bottom_bar.addWidget(exit_btn); main_layout.addLayout(bottom_bar)
        clear_btn.clicked.connect(self.clear_all_plots); exit_btn.clicked.connect(self.close)

    def _create_control_panel(self):
        panel = QGroupBox("Controls & Parameters"); layout = QGridLayout(panel)
        self.file_path_le = QLineEdit("Please select a data file..."); self.file_path_le.setReadOnly(True); browse_btn = QPushButton("Browse..."); self.column_combo = QComboBox()
        layout.addWidget(QLabel("Data File:"), 0, 0, 1, 2); layout.addWidget(self.file_path_le, 1, 0, 1, 2); layout.addWidget(browse_btn, 1, 2)
        layout.addWidget(QLabel("Signal Column:"), 2, 0, 1, 3); layout.addWidget(self.column_combo, 3, 0, 1, 3)
        self.downsample_cb = QCheckBox("Enable Downsampling"); self.downsample_cb.setChecked(True); self.downsample_sb = QSpinBox(); self.downsample_sb.setRange(1, 20); self.downsample_sb.setValue(7)
        layout.addWidget(self.downsample_cb, 4, 0, 1, 2); layout.addWidget(self.downsample_sb, 4, 2)
        source_options = ['Journal Pipeline'] + [f'DWT Q{i}' for i in range(1, 9)]
        self.hrv_source_combo = QComboBox(); self.hrv_source_combo.addItems(source_options)
        self.rr_source_combo = QComboBox(); self.rr_source_combo.addItems(source_options); self.rr_source_combo.setCurrentText("DWT Q5")
        self.vma_source_combo = QComboBox(); self.vma_source_combo.addItems(source_options); self.vma_source_combo.setCurrentText("DWT Q6")
        layout.addWidget(QLabel("HRV Metrics Source:"), 5, 0, 1, 3); layout.addWidget(self.hrv_source_combo, 6, 0, 1, 3)
        layout.addWidget(QLabel("Respiratory Rate Source:"), 7, 0, 1, 3); layout.addWidget(self.rr_source_combo, 8, 0, 1, 3)
        layout.addWidget(QLabel("Vasomotor Activity Source:"), 9, 0, 1, 3); layout.addWidget(self.vma_source_combo, 10, 0, 1, 3)
        self.run_analysis_btn = QPushButton("RUN ANALYSIS PIPELINE"); layout.addWidget(self.run_analysis_btn, 11, 0, 1, 3)
        self.results_te = QTextEdit(); self.results_te.setReadOnly(True); layout.addWidget(QLabel("Analysis Results:"), 12, 0, 1, 3); layout.addWidget(self.results_te, 13, 0, 1, 3)
        panel.setFixedWidth(450)
        browse_btn.clicked.connect(self.browse_file); self.run_analysis_btn.clicked.connect(self.run_full_analysis); self.downsample_cb.stateChanged.connect(self.downsample_sb.setEnabled)
        return panel

    def _create_scrollable_tab(self, layout):
        widget = QWidget(); widget.setLayout(layout); scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget); return scroll

    def _create_tabs(self):
        layout1 = QGridLayout(); self.raw_signal_plot, self.original_fft_plot, self.downsampled_signal_plot, self.downsampled_fft_plot, self.processed_signal_plot, self.processed_fft_plot = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        layout1.addWidget(self.raw_signal_plot, 0, 0); layout1.addWidget(self.original_fft_plot, 0, 1); layout1.addWidget(self.downsampled_signal_plot, 1, 0); layout1.addWidget(self.downsampled_fft_plot, 1, 1); layout1.addWidget(self.processed_signal_plot, 2, 0); layout1.addWidget(self.processed_fft_plot, 2, 1); self.tabs.addTab(self._create_scrollable_tab(layout1), "1. Signal Processing")
        layout_peak = QHBoxLayout(); self.peak_plot = MplCanvasWithToolbar(self); layout_peak.addWidget(self.peak_plot); self.tabs.addTab(self._create_scrollable_tab(layout_peak), "2. Peak Detection")
        layout3 = QHBoxLayout(); self.rr_tach_plot, self.rr_hist_plot = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self); layout3.addWidget(self.rr_tach_plot); layout3.addWidget(self.rr_hist_plot); self.tabs.addTab(self._create_scrollable_tab(layout3), "3. HRV - Time")
        layout4 = QHBoxLayout(); self.hrv_psd_plot, self.vaso_psd_plot = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self); layout4.addWidget(self.hrv_psd_plot); layout4.addWidget(self.vaso_psd_plot); self.tabs.addTab(self._create_scrollable_tab(layout4), "4. HRV - Freq.")
        layout5 = QHBoxLayout(); self.poincare_plot = MplCanvasWithToolbar(self); layout5.addWidget(self.poincare_plot); self.tabs.addTab(self._create_scrollable_tab(layout5), "5. HRV - Non-linear")
        layout6 = QHBoxLayout(); self.autonomic_balance_plot = MplCanvasWithToolbar(self); layout6.addWidget(self.autonomic_balance_plot); self.tabs.addTab(self._create_scrollable_tab(layout6), "6. Autonomic Balance")
        self.dwt_tabs = QTabWidget(); self.dwt_plots = {}
        for i in range(1, 9):
            tab_layout = QGridLayout(); time_plot, fft_plot, psd_plot, freq_response_plot = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
            tab_layout.addWidget(time_plot, 0, 0); tab_layout.addWidget(fft_plot, 0, 1); tab_layout.addWidget(psd_plot, 1, 0); tab_layout.addWidget(freq_response_plot, 1, 1)
            self.dwt_plots[i] = {'time': time_plot, 'fft': fft_plot, 'psd': psd_plot, 'freq_response': freq_response_plot}; self.dwt_tabs.addTab(self._create_scrollable_tab(tab_layout), f"Q{i}")
        self.tabs.addTab(self.dwt_tabs, "7. DWT Coefficients")
        layout8 = QHBoxLayout(); self.dwt_response_orig_plot, self.dwt_response_ds_plot = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self); layout8.addWidget(self.dwt_response_orig_plot); layout8.addWidget(self.dwt_response_ds_plot); self.tabs.addTab(self._create_scrollable_tab(layout8), "8. DWT Freq. Response")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PPG Data CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.file_path_le.setText(file_path)
            try:
                headers = pd.read_csv(file_path, nrows=0).columns.tolist()
                self.column_combo.clear(); self.column_combo.addItems(headers)
                if (pleth_index := next((i for i, col in enumerate(headers) if 'pleth' in col.lower()), -1)) != -1:
                    self.column_combo.setCurrentIndex(pleth_index)
            except Exception as e: self.status_bar.showMessage(f"Error reading file headers: {e}", 5000)

    def run_full_analysis(self):
        file_path = self.file_path_le.text()
        if not file_path or not os.path.exists(file_path): self.status_bar.showMessage("Error: Please select a valid CSV file.", 5000); return
        self.run_analysis_btn.setEnabled(False); self.progress_bar.show()
        self.analysis_thread = AnalysisThread(self.analyzer, file_path, self.column_combo.currentText(), self.downsample_cb.isChecked(), self.downsample_sb.value(), self.hrv_source_combo.currentText(), self.rr_source_combo.currentText(), self.vma_source_combo.currentText())
        self.analysis_thread.progress.connect(lambda p, msg: [self.progress_bar.setValue(p), self.status_bar.showMessage(msg)]); self.analysis_thread.finished.connect(self.on_analysis_finished); self.analysis_thread.error.connect(self.on_analysis_error); self.analysis_thread.start()

    def on_analysis_finished(self, results):
        self.results = results; self.update_all_plots(); self.populate_results_text()
        self.status_bar.showMessage("Analysis complete.", 5000); self.run_analysis_btn.setEnabled(True); self.progress_bar.hide()
        
    def on_analysis_error(self, message):
        self.status_bar.showMessage(f"Error: {message}", 10000); self.results_te.setText(f"ANALYSIS FAILED:\n\n{message}"); self.run_analysis_btn.setEnabled(True); self.progress_bar.hide()

    def clear_all_plots(self):
        for w in self.findChildren(MplCanvasWithToolbar): w.canvas.axes.clear(); w.canvas.apply_dark_theme(); w.canvas.draw()
        self.results_te.clear(); self.status_bar.showMessage("Cleared all plots and results.")
        
    def update_all_plots(self):
        if not self.results: return
        base, hrv, dwt_responses = self.results, self.results['hrv_results'], self.results['dwt_freq_responses']
        td, nl, freq = hrv['time_domain'], hrv['nonlinear'], hrv['frequency_domain']

        self.raw_signal_plot.canvas.plot_data([{'x': base['raw_time'], 'y': base['raw_signal'], 'title': 'Raw Signal', 'style': {'color':'#63b3ed'}}])
        self.original_fft_plot.canvas.plot_data([{'x': base['fft_original'][0], 'y': base['fft_original'][1], 'title': 'FFT of Raw Signal', 'xlabel': 'Hz', 'style': {'color':'#63b3ed'}}])
        self.downsampled_signal_plot.canvas.plot_data([{'x': base['ds_time'], 'y': base['ds_signal'], 'title': 'Downsampled Signal', 'style': {'color':'#faf089'}}])
        self.downsampled_fft_plot.canvas.plot_data([{'x': base['fft_downsampled'][0], 'y': base['fft_downsampled'][1], 'title': 'FFT of Downsampled Signal', 'xlabel': 'Hz', 'style': {'color':'#faf089'}}])
        self.processed_signal_plot.canvas.plot_data([{'x': base['preprocessed_time'], 'y': base['preprocessed_signal_main'], 'title': f"Pre-processed Signal", 'style': {'color':'#68d391'}}])
        self.processed_fft_plot.canvas.plot_data([{'x': base['fft_processed'][0], 'y': base['fft_processed'][1], 'title': 'FFT of Pre-processed Signal', 'xlabel': 'Hz', 'style': {'color':'#68d391'}}])

        peak_plot_instructions = [{'x': base['preprocessed_time'], 'y': base['preprocessed_signal_main'], 'label': 'Signal', 'title': 'Detected Cardiac Peaks & Minima', 'style': {'color':'#4299e1', 'alpha': 0.7}}]
        if (p := hrv.get('peaks')) is not None and p.size > 0: peak_plot_instructions.append({'type': 'scatter', 'x': base['preprocessed_time'][p], 'y': base['preprocessed_signal_main'][p], 'label': 'Peaks', 'style': {'marker':'x', 'color':'#fc8181', 's': 50}})
        if (m := hrv.get('minima')) is not None and m.size > 0: peak_plot_instructions.append({'type': 'scatter', 'x': base['preprocessed_time'][m], 'y': base['preprocessed_signal_main'][m], 'label': 'Minima', 'style': {'marker':'o', 'color':'#68d391', 's': 30}})
        self.peak_plot.canvas.plot_data(peak_plot_instructions)
        
        if (rr_t := hrv.get('rr_times')) is not None and len(rr_t) > 1 and (rr_i := hrv.get('rr_intervals_s')) is not None: self.rr_tach_plot.canvas.plot_data([{'x': rr_t[1:], 'y': rr_i * 1000, 'title': 'RR Tachogram', 'xlabel':'Time (s)', 'ylabel':'RR (ms)', 'style': {'color':'#faf089', 'marker':'.', 'linestyle':'-'}}])
        if (hist_c := td['rr_histogram'][0]).size > 1: bin_w = np.diff(td['rr_histogram'][1])[0]; self.rr_hist_plot.canvas.plot_data([{'x': td['rr_histogram'][1][:-1] + bin_w / 2, 'y': hist_c, 'type': 'bar', 'title': 'RR Histogram', 'xlabel':'RR (ms)', 'ylabel':'Count', 'style': {'color':'#f6ad55', 'width': bin_w}}])

        if freq and freq.get('psd_freqs') is not None and len(freq['psd_freqs']) > 0: self.hrv_psd_plot.canvas.plot_data([{'x': freq['psd_freqs'], 'y': freq['psd_values'], 'title': "HRV Power Spectrum", 'xlabel':'Hz', 'ylabel':'PSD (s²/Hz)'}, {'type': 'axvline', 'x': 0.04, 'label': 'VLF/LF', 'style': {'color':'#f6ad55', 'linestyle':'--'}}, {'type': 'axvline', 'x': 0.15, 'label': 'LF/HF', 'style': {'color':'#63b3ed', 'linestyle':'--'}}])
        
        vma_psd_f, vma_psd_v = base['vasomotor_psd']
        self.vaso_psd_plot.canvas.plot_data([{'x': vma_psd_f, 'y': vma_psd_v, 'title': f"PSD of Vasomotor Source ({self.vma_source_combo.currentText()})", 'xlabel': 'Hz', 'ylabel': 'PSD', 'label': 'PSD Signal', 'style': {'color': '#a0aec0'}}, {'type': 'fill_between', 'x': vma_psd_f, 'y1': vma_psd_v, 'y2': 0, 'label': 'VMA Band (0.04-0.15Hz)', 'style': {'where': (vma_psd_f >= 0.04) & (vma_psd_f <= 0.15), 'color': '#f6ad55', 'alpha': 0.6}}])
        
        if len(nl['poincare_x']) > 1:
            mean_x, mean_y, sd1, sd2 = np.mean(nl['poincare_x']), np.mean(nl['poincare_y']), nl['sd1'], nl['sd2']
            min_rr, max_rr = min(np.min(nl['poincare_x']), np.min(nl['poincare_y']))*0.95, max(np.max(nl['poincare_x']), np.max(nl['poincare_y']))*1.05
            cos_a, sin_a = np.cos(np.pi/4), np.sin(np.pi/4)
            self.poincare_plot.canvas.plot_data([{'type': 'scatter', 'x': nl['poincare_x'], 'y': nl['poincare_y'], 'title': 'Poincaré Plot', 'xlabel':'RRn (ms)', 'ylabel':'RRn+1 (ms)', 'style': {'color':'#81e6d9', 's': 5, 'alpha': 0.8}}, {'x': [min_rr, max_rr], 'y': [min_rr, max_rr], 'label': 'Line of Identity', 'style': {'color':'#a0aec0', 'linestyle':'--'}}, {'x': [mean_x-sd1*cos_a, mean_x+sd1*cos_a], 'y': [mean_y+sd1*sin_a, mean_y-sd1*sin_a], 'label': f'SD1: {sd1:.2f} ms', 'style': {'color':'#f6ad55', 'linewidth': 3}}, {'x': [mean_x-sd2*cos_a, mean_x+sd2*cos_a], 'y': [mean_y-sd2*sin_a, mean_y+sd2*sin_a], 'label': f'SD2: {sd2:.2f} ms', 'style': {'color':'#63b3ed', 'linewidth': 3}}])

        rmssd, lf_hf = td.get('rmssd', 0), freq.get('lf_hf_ratio', 0)
        if np.isinf(lf_hf) or lf_hf > 10: lf_hf = 10
        ax = self.autonomic_balance_plot.canvas.axes; ax.clear(); xlim, ylim = [0, max(4, lf_hf*1.5)], [0, max(80, rmssd*1.5)]
        ax.axhspan(40, ylim[1], fc=mcolors.to_rgba('#68D391',.15)); ax.axhspan(0, 40, fc=mcolors.to_rgba('#4299E1',.15)); ax.axvspan(1.5, xlim[1], fc=mcolors.to_rgba('#DD6B20',.2), hatch='//')
        ax.plot(lf_hf, rmssd, 'o', ms=10, color='#f56565', zorder=10); ax.axvline(x=1.5, color='#a0aec0', ls='--'); ax.axhline(y=40, color='#a0aec0', ls='--'); ax.set(title="Autonomic Balance Diagram", xlabel="Sympathetic (LF/HF Ratio)", ylabel="Parasympathetic (RMSSD ms)", xlim=xlim, ylim=ylim)
        ax.text(.95, .95, 'High Stress', transform=ax.transAxes, ha='right', va='top', color='#f6ad55'); ax.text(.05, .95, 'Healthy/Relaxed', transform=ax.transAxes, ha='left', va='top', color='#68d391'); ax.text(.05, .05, 'Fatigue', transform=ax.transAxes, ha='left', va='bottom', color='#63b3ed'); ax.text(.95, .05, 'High Stress & Fatigue', transform=ax.transAxes, ha='right', va='bottom', color='#e53e3e')
        self.autonomic_balance_plot.canvas.apply_dark_theme(); self.autonomic_balance_plot.canvas.draw()

        for i in range(1, 9):
            if i in base['dwt_results'] and (resp := dwt_responses.get(i, {}).get('ds')):
                dwt_sig = base['dwt_results'][i]; fft_f, fft_m = self.analyzer.fft_magnitude_and_frequencies(dwt_sig); psd_f, psd_v = welch_from_scratch(dwt_sig, base['downsampled_fs'], fft_func=self.analyzer.fft_from_scratch)
                self.dwt_plots[i]['time'].canvas.plot_data([{'x': base['preprocessed_time'], 'y': dwt_sig, 'title': f'DWT Q{i} Signal', 'style': {'color':'#faf089'}}])
                self.dwt_plots[i]['fft'].canvas.plot_data([{'x': fft_f, 'y': fft_m, 'title': f'FFT of Q{i}', 'xlabel':'Hz', 'style': {'color':'#f6ad55'}}])
                self.dwt_plots[i]['psd'].canvas.plot_data([{'x': psd_f, 'y': psd_v, 'title': f'PSD of Q{i}', 'xlabel':'Hz', 'ylabel':'PSD', 'style': {'color':'#81e6d9'}}])
                # [PLOT FIX] Use correct label and no y-limits for linear magnitude plot
                self.dwt_plots[i]['freq_response'].canvas.plot_data([{'x': resp[0], 'y': resp[1], 'title': f'Q{i} Filter Response (DS)', 'xlabel':'Hz', 'ylabel':'Magnitude', 'style': {'color':'#f6ad55'}}])
        
        colors = plt.cm.viridis(np.linspace(0, 1, 8))
        orig_resp = [{'x':r['orig'][0], 'y':r['orig'][1], 'label':f'Q{i}', 'style':{'color':colors[i-1]}} for i,r in dwt_responses.items() if r.get('orig')]
        if orig_resp:
            # [PLOT FIX] Use correct label and no y-limits
            orig_resp[0].update({'title':f"Filter Response @ Original FS ({base['original_fs']:.1f} Hz)", 'xlabel':"Hz", 'ylabel':"Magnitude"})
            self.dwt_response_orig_plot.canvas.plot_data(orig_resp)
        ds_resp = [{'x':r['ds'][0], 'y':r['ds'][1], 'label':f'Q{i}', 'style':{'color':colors[i-1]}} for i,r in dwt_responses.items() if r.get('ds')]
        if ds_resp:
            # [PLOT FIX] Use correct label and no y-limits
            ds_resp[0].update({'title':f"Filter Response @ Downsampled FS ({base['downsampled_fs']:.1f} Hz)", 'xlabel':"Hz", 'ylabel':"Magnitude"})
            self.dwt_response_ds_plot.canvas.plot_data(ds_resp)

        self.status_bar.showMessage("All plots updated.")

    def populate_results_text(self):
        if not self.results: return
        td, fd, nl = self.results['hrv_results']['time_domain'], self.results['hrv_results']['frequency_domain'], self.results['hrv_results']['nonlinear']

        text = "PHYSIOLOGICAL RATE ESTIMATION\n---------------------------------\n"
        text += f"{'Respiratory Rate:':<25}{self.results.get('respiratory_rate_brpm', 0):>10.2f} brpm (from {self.rr_source_combo.currentText()})\n"
        text += f"{'Vasomotor Activity:':<25}{self.results.get('vasomotor_rate_cpm', 0):>10.2f} cpm (from {self.vma_source_combo.currentText()})\n\n"

        text += f"HRV ANALYSIS (from {self.hrv_source_combo.currentText()})\n\n"
        text += "TIME DOMAIN\n---------------------------------\n"
        text += f"{'Mean HR:':<25}{td.get('mean_hr', 0):>10.2f} bpm\n"
        text += f"{'SDNN:':<25}{td.get('sdnn', 0):>10.2f} ms\n"
        text += f"{'SDANN (5min):':<25}{td.get('sdann', 0):>10.2f} ms\n"
        text += f"{'SDNN Index (5min):':<25}{td.get('sdnn_index', 0):>10.2f} ms\n"
        text += f"{'RMSSD:':<25}{td.get('rmssd', 0):>10.2f} ms\n"
        text += f"{'SDSD:':<25}{td.get('sdsd', 0):>10.2f} ms\n"
        text += f"{'NN50:':<25}{td.get('nn50', 0):>10.0f}\n"
        text += f"{'pNN50:':<25}{td.get('pnn50', 0):>10.2f} %\n"
        text += f"{'HRV Tri Index (HTI):':<25}{td.get('hti', 0):>10.2f}\n"
        text += f"{'TINN:':<25}{td.get('tinn', 0):>10.2f} ms\n"
        text += f"{'CVNN:':<25}{td.get('cvnn', 0):>10.4f}\n"
        text += f"{'CVSD:':<25}{td.get('cvsd', 0):>10.4f}\n"
        text += f"{'Skewness of NN:':<25}{td.get('skewness', 0):>10.4f}\n\n"
        
        text += "FREQUENCY DOMAIN\n---------------------------------\n"
        text += f"{'Total Power (TP):':<25}{fd.get('total_power', 0):>10.2f} s²\n"
        text += f"{'Total Power of LF:':<25}{fd.get('lf_power', 0):>10.2f} s²\n"
        text += f"{'Total Power of HF:':<25}{fd.get('hf_power', 0):>10.2f} s²\n"
        text += f"{'LF/HF Ratio:':<25}{fd.get('lf_hf_ratio', 0):>10.2f}\n"
        text += f"{'LF (n.u.):':<25}{fd.get('lf_nu', 0):>10.2f} n.u.\n"
        text += f"{'HF (n.u.):':<25}{fd.get('hf_nu', 0):>10.2f} n.u.\n"
        text += f"{'Peak Frequency of LF:':<25}{fd.get('peak_lf', 0):>10.3f} Hz\n"
        text += f"{'Peak Frequency of HF:':<25}{fd.get('peak_hf', 0):>10.3f} Hz\n\n"
        
        text += "NON-LINEAR METHODS\n---------------------------------\n"
        text += f"{'SD1:':<25}{nl.get('sd1', 0):>10.2f} ms\n"
        text += f"{'SD2:':<25}{nl.get('sd2', 0):>10.2f} ms\n"
        text += f"{'SD1/SD2 Ratio:':<25}{nl.get('sd1_sd2_ratio', 0):>10.2f}\n"
        self.results_te.setText(text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())