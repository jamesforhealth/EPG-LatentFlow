import sys
import os
import json
from natsort import natsorted
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QInputDialog, QSlider, QFileDialog, QTabWidget, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QSplitter, QCheckBox, QLineEdit, QTextEdit, QToolTip
from PyQt5.QtGui import QMouseEvent, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import pyqtgraph as pg
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample
from influxDB_downloader import get_user_sessions, get_session_data, format_timestamp, sanitize_filename
from model_find_peaks import detect_peaks_from_signal
from model_pulse_representation import EPGBaselinePulseAutoencoder#, predict_reconstructed_signal
from model_pulse_representation_explainable import DisentangledAutoencoder, predict_reconstructed_signal, predict_corrected_reconstructed_signal
import torch
import scipy
# from model_wearing_anomaly_detection import predict_reconstructed_signal, predict_reconstructed_signal2, predict_reconstructed_signal_pulse
# from model_pulse_representation import predict_LSTM_reconstructed_signal
from model_wearing_anomaly_detection_transformer import predict_transformer_reconstructed_signal
def find_peaks_helper(data, sample_rate, drop_rate, drop_rate_gain, timer_init, timer_peak_refractory_period, peak_refinement_window, Vpp_method, Vpp_threshold):
    peaks_top = []
    peaks_bottom = []
    prev_sample_top = data[0]
    prev_sample_bottom = data[0] * -1.0
    rising_top = False
    rising_bottom = False
    timer_top = timer_init
    timer_bottom = timer_init
    timer_peak_refractory_period_top = timer_peak_refractory_period
    timer_peak_refractory_period_bottom = timer_peak_refractory_period
    envelope_top = data[0]
    envelope_bottom = data[0] * -1.0
    envelope_plot_top = [envelope_top]
    envelope_plot_bottom = [envelope_bottom]
    Vpp_too_small = False
    peak_refinement_offset = int(round(peak_refinement_window * sample_rate))

    for i in range(1 + peak_refinement_offset, len(data) - peak_refinement_offset):
        curr_point_top = data[i]
        curr_point_bottom = curr_point_top * -1.0

        if curr_point_top > prev_sample_top:
            rising_top = True
            envelope_top -= (drop_rate * drop_rate_gain / sample_rate) if timer_top <= 0 else (drop_rate / sample_rate)
        elif curr_point_top < prev_sample_top and rising_top:
            if prev_sample_top >= envelope_top and not Vpp_too_small and timer_peak_refractory_period_top <= 0:
                peaks_top.append(i - 1)
                rising_top = False
                envelope_top = curr_point_top
                envelope_plot_top.pop()
                envelope_plot_top.append(prev_sample_top)
                timer_top = timer_init
                timer_peak_refractory_period_top = timer_peak_refractory_period
            else:
                envelope_top -= (drop_rate * drop_rate_gain / sample_rate) if timer_top <= 0 else (drop_rate / sample_rate)
        else:
            envelope_top -= (drop_rate * drop_rate_gain / sample_rate) if timer_top <= 0 else (drop_rate / sample_rate)

        if envelope_top < curr_point_top:
            envelope_top = curr_point_top
            rising_top = True

        prev_sample_top = curr_point_top
        timer_top -= 1.0 / sample_rate
        timer_peak_refractory_period_top -= 1.0 / sample_rate
        envelope_plot_top.append(envelope_top)

        if curr_point_bottom > prev_sample_bottom:
            rising_bottom = True
            envelope_bottom -= (drop_rate * drop_rate_gain / sample_rate) if timer_bottom <= 0 else (drop_rate / sample_rate)
        elif curr_point_bottom < prev_sample_bottom and rising_bottom:
            if prev_sample_bottom >= envelope_bottom and not Vpp_too_small and timer_peak_refractory_period_bottom <= 0:
                peaks_bottom.append(i - 1)
                rising_bottom = False
                envelope_bottom = curr_point_bottom
                envelope_plot_bottom.pop()
                envelope_plot_bottom.append(prev_sample_bottom)
                timer_bottom = timer_init
                timer_peak_refractory_period_bottom = timer_peak_refractory_period
            else:
                envelope_bottom -= (drop_rate * drop_rate_gain / sample_rate) if timer_bottom <= 0 else (drop_rate / sample_rate)
        else:
            envelope_bottom -= (drop_rate * drop_rate_gain / sample_rate) if timer_bottom <= 0 else (drop_rate / sample_rate)

        if envelope_bottom < curr_point_bottom:
            envelope_bottom = curr_point_bottom
            rising_bottom = True

        prev_sample_bottom = curr_point_bottom
        timer_bottom -= 1.0 / sample_rate
        timer_peak_refractory_period_bottom -= 1.0 / sample_rate
        envelope_plot_bottom.append(envelope_bottom)

        Vpp = envelope_top - envelope_bottom * -1
        Vpp_too_small = Vpp <= Vpp_threshold

    envelope_plot_bottom = [-x for x in envelope_plot_bottom]
    Vpp_plot = []

    if Vpp_method == "continuous":
        Vpp_plot = [x - y for x, y in zip(envelope_plot_top, envelope_plot_bottom)]
    elif Vpp_method == "on_peak":
        curr_top = envelope_plot_top[0]
        curr_bottom = envelope_plot_bottom[0]
        for i in range(len(envelope_plot_top)):
            if i in peaks_top:
                curr_top = envelope_plot_top[i]
            if i in peaks_bottom:
                curr_bottom = envelope_plot_bottom[i]
            Vpp_plot.append(curr_top - curr_bottom)
    else:
        Vpp_plot = [x - y for x, y in zip(envelope_plot_top, envelope_plot_bottom)]

    zero_pad = [0.0] * peak_refinement_offset
    envelope_plot_top = zero_pad + envelope_plot_top
    envelope_plot_bottom = zero_pad + envelope_plot_bottom

    return peaks_top, peaks_bottom, envelope_plot_top, envelope_plot_bottom, Vpp_plot

def findValleysBasedOnPeaks(data, peaks, sample_rate):
    valleys = []
    searchRange = int((15 * sample_rate) / 100)
    # print(f'searchRange:{searchRange}, peaks:{peaks}, sample_rate:{sample_rate}')

    for peak in peaks:
        foundValley = False

        for i in range(1, searchRange + 1):
            if peak + i + 1 >= len(data):
                break
            if data[peak + i] <= data[peak + i - 1] and data[peak + i] <= data[peak + i + 1]:
                valleys.append(peak + i)
                foundValley = True
                break

        if not foundValley:
            valleys.append(peak + int((10 * sample_rate) / 100))  # 如果没有找到valley，就用peak+0.1秒的sample点的位置视为valley
    return valleys

def find_next_valley(peak, valleys):
    for i, valley in enumerate(valleys):
        if valley > peak:
            return i
    return -1  # Return -1 if no next valley is found

def find_epg_points(input_data, peaks, valleys, sample_rate):
    results = []
    peaks_num = len(peaks)
    
    for i in range(peaks_num - 1):
        peak_idx = peaks[i]
        nextpeak_idx = peaks[i + 1]
        peak_len = nextpeak_idx - peak_idx
        next_valley_idx = find_next_valley(peaks[i], valleys)
        if next_valley_idx == -1 or valleys[next_valley_idx] >= peaks[i + 1]:
            results.append([])  # If no valid valley is found, add an empty result
            continue

        start_idx = valleys[next_valley_idx]+2  # The start index of the EPG is the valley + 2 samples
        max_search_idx = int(peak_idx + 0.6 * peak_len)  # Only search within 60% of the wave length
        # print(f'Segment of signal peak_idx: {peak_idx}, nextpeak_idx: {nextpeak_idx}, start_idx: {start_idx}, max_search_idx: {max_search_idx}')

        state = 'search_max'  # Start by searching for a local maximum after the valley
        found_points = []
        
        for j in range(start_idx, min(nextpeak_idx, max_search_idx)):
            if state == 'search_max' and input_data[j] >= input_data[j + 1]:
                found_points.append((j, -1))  # Found a local maximum
                state = 'search_min'
            elif state == 'search_min' and input_data[j] <= input_data[j + 1]:
                found_points[-1] = (found_points[-1][0], j)  # Found a local minimum and pair it with the last maximum
                if len(found_points) == 2:
                    break  # We have found two sets of extrema
                state = 'search_max'
    
        systolica_range = 0.25 * sample_rate
        # print(f'diff1: {diff1}, diff2: {diff2}')
        if len(found_points) == 0:
            found_points = [(-1, -1),(-1, -1)]
        elif len(found_points) == 1:
            if found_points[0][0] - peak_idx <= systolica_range: #找到的那一組是za點
                found_points = [found_points[0],(-1, -1)]
            else: #找到的那一组是bc点
                found_points = [(-1, -1), found_points[0]]
        elif found_points[0][0] - peak_idx > systolica_range: #找到不少於兩組但第一組是bc點
            found_points = [(-1, -1), found_points[0]]
        
        elif found_points[1][0] - peak_idx <= systolica_range: #找到不少於兩組但第二組還不是bc點
            found_points = [found_points[0], found_points[2]] if len(found_points) > 2 else [found_points[0], (-1, -1)]  #如果za附近多於一组则会出现例外，但不好处理    


        results.append(found_points)

    return results

def gaussian_smooth(input, window_size, sigma):
    half_window = window_size // 2
    output = np.zeros_like(input)
    #weights = np.zeros(window_size)
    weights = np.zeros(2 * half_window + 1)
    weight_sum = 0

    # Calculate Gaussian weights
    for i in range(-half_window, half_window + 1):
        weights[i + half_window] = np.exp(-0.5 * (i / sigma) ** 2)
        weight_sum += weights[i + half_window]

    # Normalize weights
    weights /= weight_sum

    # Apply Gaussian smoothing
    for i in range(len(input)):
        smoothed_value = 0
        for j in range(-half_window, half_window + 1):
            index = i + j
            if 0 <= index < len(input):
                smoothed_value += input[index] * weights[j + half_window]
        output[i] = smoothed_value

    # Copy border values from the input
    output[:window_size] = input[:window_size]
    output[-window_size:] = input[-window_size:]

    return output

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseline_model3 = EPGBaselinePulseAutoencoder(100).to(self.device)
        model_path = 'pulse_interpolate_autoencoder.pth'
        self.baseline_model3.load_state_dict(torch.load(model_path,map_location = self.device))
        self.baseline_model3.eval()

        self.disentangled_model = DisentangledAutoencoder(target_len=100, physio_dim=15, wear_dim=10).to(self.device)
        model_path = './DisentangledAutoencoder_pretrain_wearing2.pth'
        self.disentangled_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.disentangled_model.eval()

        self.setWindowTitle("Waveform Labeling")
        self.resize(1200, 800)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setClipToView(True)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Sample Points Index')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')

        self.legend = pg.LegendItem()
        self.legend.setParentItem(self.plot_widget.graphicsItem())

        self.raw_data = []
        self.smoothed_data = []
        self.standardized_data = []
        self.smoothed_data_100hz = []
        self.reconstructed_signal = []
        self.wear_corrected_signal = []
        self.noisy_data = []
        self.target_snr_db = 20
        self.x_points = []
        self.y_points = []
        self.z_points = []
        self.a_points = []
        self.b_points = []
        self.c_points = []
        self.dataType = ""
        self.macaddress = ""
        self.anomaly_list = []
        self.selected_start_peak_idx = None
        self.selected_end_peak_idx = None
        self.nearest_peak_idx = None
        self.sample_rate = 100
        self.selected_points = []
        self.current_relative_path = ''
        self.fft_result = np.array([])      
        self.fft_freq = np.array([])
        self.latent_vectors = []

        self.selected_points_label = QLabel("Selected Points: []")

        self.file_tree_widget = QTreeWidget()
        self.file_tree_widget.setHeaderLabels(["DB Files"])
        self.file_tree_widget.itemClicked.connect(self.load_selected_file)

        self.user_id_edit = QLineEdit()
        self.user_id_edit.setPlaceholderText("Enter User ID")

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_user_data)

        self.db_folder = "DB"
        self.labeled_db_folder = "labeled_DB"

        file_count = self.count_files_in_directory(self.db_folder, ".json")
        labeled_file_count = self.count_files_in_directory(self.labeled_db_folder, ".json")
        file_count_label = QLabel(f"labeled/db: {labeled_file_count}/{file_count}")
        
        user_id_layout = QHBoxLayout()
        user_id_layout.addWidget(QLabel("User ID:"))
        user_id_layout.addWidget(self.user_id_edit)
        user_id_layout.addWidget(self.reload_button)
        user_id_layout.addWidget(file_count_label)

        self.db_combo_box = QComboBox()
        self.db_combo_box.addItem("DB")
        self.db_combo_box.addItem("labeled_DB")
        self.db_combo_box.addItem("wearing_consistency")
        self.db_combo_box.currentIndexChanged.connect(self.update_file_tree)

        file_tree_layout = QVBoxLayout()
        file_tree_layout.addWidget(self.db_combo_box)
        file_tree_layout.addLayout(user_id_layout)
        file_tree_layout.addWidget(self.file_tree_widget)

        self.checkbox_raw_data = QCheckBox("Raw Data")
        self.checkbox_raw_data.setChecked(False)
        self.checkbox_raw_data.stateChanged.connect(self.plot_data)

        self.checkbox_smoothed_data = QCheckBox("Smoothed Data")
        self.checkbox_smoothed_data.setChecked(True)
        self.checkbox_smoothed_data.stateChanged.connect(self.plot_data)

        self.checkbox_reconstructed_signal = QCheckBox("Reconstructed Data")
        self.checkbox_reconstructed_signal.setChecked(True)
        self.checkbox_reconstructed_signal.stateChanged.connect(self.plot_data)

        self.checkbox_wear_corrected_signal = QCheckBox("Wear Corrected Signal")
        self.checkbox_wear_corrected_signal.setChecked(True)
        self.checkbox_wear_corrected_signal.stateChanged.connect(self.plot_data)

        self.checkbox_noisy_data = QCheckBox("Noisy Data")
        self.checkbox_noisy_data.setChecked(False)
        self.checkbox_noisy_data.stateChanged.connect(self.plot_data)

        self.resample_noise_button = QPushButton("Resample Noise")
        self.resample_noise_button.clicked.connect(self.resample_noise)

        self.checkbox_x_points = QCheckBox("X Points")
        self.checkbox_x_points.setChecked(True)
        self.checkbox_x_points.stateChanged.connect(self.plot_data)

        self.checkbox_y_points = QCheckBox("Y Points")
        self.checkbox_y_points.setChecked(False)
        self.checkbox_y_points.stateChanged.connect(self.plot_data)

        self.checkbox_z_points = QCheckBox("Z Points")
        self.checkbox_z_points.setChecked(False)
        self.checkbox_z_points.stateChanged.connect(self.plot_data)

        self.checkbox_a_points = QCheckBox("A Points")
        self.checkbox_a_points.setChecked(False)
        self.checkbox_a_points.stateChanged.connect(self.plot_data)

        self.checkbox_b_points = QCheckBox("B Points")
        self.checkbox_b_points.setChecked(False)
        self.checkbox_b_points.stateChanged.connect(self.plot_data)

        self.checkbox_c_points = QCheckBox("C Points")
        self.checkbox_c_points.setChecked(False)
        self.checkbox_c_points.stateChanged.connect(self.plot_data)



        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.checkbox_raw_data)
        checkbox_layout.addWidget(self.checkbox_smoothed_data)
        checkbox_layout.addWidget(self.checkbox_noisy_data)
        checkbox_layout.addWidget(self.checkbox_reconstructed_signal)
        checkbox_layout.addWidget(self.checkbox_wear_corrected_signal)
        checkbox_layout.addWidget(self.checkbox_x_points)
        checkbox_layout.addWidget(self.checkbox_y_points)
        checkbox_layout.addWidget(self.checkbox_z_points)
        checkbox_layout.addWidget(self.checkbox_a_points)
        checkbox_layout.addWidget(self.checkbox_b_points)
        checkbox_layout.addWidget(self.checkbox_c_points)
        checkbox_layout.addWidget(self.resample_noise_button)

        self.x_points_edit = QLineEdit()
        self.x_points_edit.setPlaceholderText("X Points")
        self.x_points_edit.textChanged.connect(self.update_x_points_from_edit)

        self.y_points_edit = QLineEdit()
        self.y_points_edit.setPlaceholderText("Y Points")
        self.y_points_edit.textChanged.connect(self.update_y_points_from_edit)

        self.z_points_edit = QLineEdit()
        self.z_points_edit.setPlaceholderText("Z Points")
        self.z_points_edit.textChanged.connect(self.update_z_points_from_edit)

        self.a_points_edit = QLineEdit()
        self.a_points_edit.setPlaceholderText("A Points")
        self.a_points_edit.textChanged.connect(self.update_a_points_from_edit)

        self.b_points_edit = QLineEdit()
        self.b_points_edit.setPlaceholderText("B Points")
        self.b_points_edit.textChanged.connect(self.update_b_points_from_edit)

        self.c_points_edit = QLineEdit()
        self.c_points_edit.setPlaceholderText("C Points")
        self.c_points_edit.textChanged.connect(self.update_c_points_from_edit)

        points_layout = QVBoxLayout()

        # 添加新的UI元素
        self.pulse_index_label = QLabel("Pulse: N/A")
        self.latent_vector_display = QTextEdit()
        self.latent_vector_display.setReadOnly(True)
        self.latent_vector_display.setLineWrapMode(QTextEdit.WidgetWidth)
        self.pulse_slider = QSlider(Qt.Horizontal)
        self.pulse_slider.valueChanged.connect(self.update_latent_vector_display)

        # 修改布局
        # 在现有的 UI 元素之后添加
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(20)
        self.amplitude_slider.setMaximum(500)
        self.amplitude_slider.setValue(100)
        self.amplitude_slider.valueChanged.connect(self.update_amplitude)

        self.amplitude_label = QLabel("Amplitude: 1.00x")

        points_layout.addWidget(QLabel("Amplitude:"))
        points_layout.addWidget(self.amplitude_slider)
        points_layout.addWidget(self.amplitude_label)

        points_layout.addWidget(QLabel("Selected Pulse:"))
        points_layout.addWidget(self.pulse_slider)
        points_layout.addWidget(self.pulse_index_label)
        points_layout.addWidget(QLabel("Latent Vector:"))
        points_layout.addWidget(self.latent_vector_display)

        self.anomaly_text_edit = QTextEdit()
        self.anomaly_text_edit.setReadOnly(True)
        points_layout.addWidget(QLabel("Anomaly Segment Labeling (按住Control鍵之後用左鍵依序選取頭尾的peak點或是整筆量測的首尾點):"))

        anomaly_text_layout = QHBoxLayout()
        anomaly_text_layout.addWidget(self.anomaly_text_edit)

        self.clear_anomaly_button = QPushButton("Clear")
        self.clear_anomaly_button.clicked.connect(self.clear_anomaly_labels)
        anomaly_text_layout.addWidget(self.clear_anomaly_button)
        points_layout.addLayout(anomaly_text_layout)
        
        self.recalculate_y_button = QPushButton("Recalculate Y Points")
        self.recalculate_y_button.clicked.connect(self.recalculate_y_points)
        points_layout.addWidget(self.recalculate_y_button)

        self.recalculate_zabc_button = QPushButton("Recalculate ZABC Points")
        self.recalculate_zabc_button.clicked.connect(self.recalculate_zabc_points)
        points_layout.addWidget(self.recalculate_zabc_button)

        points_layout.addWidget(self.selected_points_label)
        points_layout.addWidget(QLabel("Points Index:"))
        points_layout.addWidget(QLabel("X Points:"))
        points_layout.addWidget(self.x_points_edit)
        points_layout.addWidget(QLabel("Y Points:"))
        points_layout.addWidget(self.y_points_edit)
        points_layout.addWidget(QLabel("Z Points:"))
        points_layout.addWidget(self.z_points_edit)
        points_layout.addWidget(QLabel("A Points:"))
        points_layout.addWidget(self.a_points_edit)
        points_layout.addWidget(QLabel("B Points:"))
        points_layout.addWidget(self.b_points_edit)
        points_layout.addWidget(QLabel("C Points:"))
        points_layout.addWidget(self.c_points_edit)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_labeled_data)
        points_layout.addWidget(self.save_button)

        time_domain_layout = QHBoxLayout()
        time_domain_layout.addWidget(self.plot_widget)
        time_domain_layout.addLayout(checkbox_layout)

        self.time_domain_widget = QWidget()
        self.time_domain_widget.setLayout(time_domain_layout)

        self.init_fft_tab()

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.time_domain_widget, "Time Domain")
        self.tab_widget.addTab(self.fft_tab, "FFT")

        main_layout = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(QWidget())

        main_layout.widget(1).setLayout(points_layout)

        main_layout.setStretchFactor(0, 4)
        main_layout.setStretchFactor(1, 1)

        central_widget = QSplitter(Qt.Horizontal)
        central_widget.addWidget(QWidget())
        central_widget.addWidget(QWidget())

        central_widget.widget(0).setLayout(file_tree_layout)
        central_widget.widget(1).setLayout(QVBoxLayout())
        central_widget.widget(1).layout().addWidget(main_layout)

        central_widget.setStretchFactor(0, 1)
        central_widget.setStretchFactor(1, 4)

        self.setCentralWidget(central_widget)

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)

        self.load_db_files()

        self.floating_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(color=(255, 0, 0), width=2), brush=pg.mkBrush(255, 255, 255, 120))
        self.plot_widget.addItem(self.floating_marker)

    def update_amplitude(self):
        amplitude_factor = self.amplitude_slider.value() / 100
        self.amplitude_label.setText(f"Amplitude: {amplitude_factor:.2f}x")
        self.update_latent_vector_display()

    def get_current_pulse(self):
        index = self.pulse_slider.value()
        if index < len(self.x_points) - 1:
            start = self.x_points[index]
            end = self.x_points[index + 1]
            return self.standardized_data[start:end]
        return None
    
    def calculate_latent_vector(self, pulse):
        if len(pulse) > 1:
            
            # 标准化
            # pulse_normalized = (pulse - np.mean(pulse)) / np.std(pulse)
            
            target_len = 100
            interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
            pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, target_len))
            pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, latent_vector = self.baseline_model3(pulse_tensor)
            
            return latent_vector.squeeze().cpu().numpy()
        return None

    def init_fft_tab(self):
        self.fft_plot_widget = pg.PlotWidget()
        self.fft_plot_widget.setLabel('left', 'Magnitude')
        self.fft_plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.fft_plot_widget.scene().sigMouseMoved.connect(self.on_fft_mouse_move)
        
        self.fft_floating_label = pg.TextItem(text="", color=(0, 0, 0))
        self.fft_plot_widget.addItem(self.fft_floating_label)

        self.window_size = 2048
        self.window_start = 0

        self.window_start_slider = QSlider(Qt.Horizontal)
        self.window_start_slider.setMinimum(0)
        self.window_start_slider.setMaximum(0)
        self.window_start_slider.valueChanged.connect(self.update_fft_plot)
        self.window_start_label = QLabel("Window Start: 0.00s")

        fft_layout = QVBoxLayout()
        fft_layout.addWidget(self.fft_plot_widget)
        fft_layout.addWidget(QLabel("Window Start:"))
        fft_layout.addWidget(self.window_start_slider)
        fft_layout.addWidget(self.window_start_label)

        self.fft_tab = QWidget()
        self.fft_tab.setLayout(fft_layout)
        
    def count_files_in_directory(self, directory, file_extension):
        total_count = 0
        icp_count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_extension):
                    total_count += 1
                    if "icp" in file.lower() or "ccp" in file.lower():
                      icp_count += 1
        return total_count,icp_count
    
    def on_fft_mouse_move(self, event):
        if len(self.fft_freq) == 0:
            return
        pos = event
        if self.fft_plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.fft_plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            nearest_idx = np.abs(self.fft_freq[:len(self.fft_freq)//2] - x).argmin()
            nearest_freq = self.fft_freq[nearest_idx]
            nearest_mag = np.abs(self.fft_result[nearest_idx])
            
            self.fft_floating_label.setText(f"Freq: {nearest_freq:.2f} Hz, Mag: {nearest_mag:.2f}")
            self.fft_floating_label.setPos(x, y)
        else:
            self.fft_floating_label.setText("")

    def reload_user_data(self):
        user_id = self.user_id_edit.text()
        if user_id:
            user_dir = os.path.join("DB", str(user_id))
            os.makedirs(user_dir, exist_ok=True)  # 為用戶建立資料夾

            sessions = get_user_sessions(user_id)
            # print(f'sessions: {sessions}')
            if sessions is not None:
                for session in sessions["timestamp"]:
                    timestamp = session
                    note = sessions["session_data"]["session_notes"][sessions["timestamp"].index(timestamp)]
                    macaddress = sessions["session_data"]["BLE_MAC_ADDRESS"][sessions["timestamp"].index(timestamp)]
                    file_name = f"({format_timestamp(timestamp)}),({sanitize_filename(note)}).json"
                    print(f'file_name: {file_name}')
                    file_path = os.path.join(user_dir, file_name)

                    if not os.path.exists(file_path):
                        session_data = get_session_data(user_id, timestamp, macaddress)
                        if session_data is not None:
                            session_info = {
                                "user_id": user_id,
                                "timestamp": timestamp,
                                "macaddress": macaddress,
                                "sample_rate": sessions["session_data"]["sample_rate"][sessions["timestamp"].index(timestamp)],
                                "session_note": note,
                                "raw_data": session_data["data"]["data"]
                            }
                            with open(file_path, "w") as file:
                                json.dump(session_info, file)
                            print(f'session_info: {session_info}')
                            print(f"Data saved to {file_path}")
                        else:
                            print("Failed to retrieve data")

            self.load_db_files()

    def find_nearest_peak(self, x):
        if x == 0:
            return -1
        elif x == len(self.smoothed_data) - 1:
            return len(self.x_points)
        else:
            nearest_peak_idx = None
            min_distance = float('inf')
            for peak_idx in range(len(self.x_points)):
                peak_x = self.x_points[peak_idx]
                distance = abs(x - peak_x)
                if distance < min_distance:
                    min_distance = distance
                    nearest_peak_idx = peak_idx
            return nearest_peak_idx

    def highlight_peak(self, peak_idx):
        if hasattr(self, 'peak_highlight'):
            self.plot_widget.removeItem(self.peak_highlight)
        if peak_idx == -1:
            x = 0
        elif peak_idx == len(self.x_points):
            x = len(self.smoothed_data) - 1
        else:
            x = self.x_points[peak_idx]
        self.peak_highlight = self.plot_widget.plot([x], [self.smoothed_data[x]], pen=None, symbol='o', symbolSize=10, symbolBrush=(255, 0, 0))

    def clear_peak_highlight(self):
        if hasattr(self, 'peak_highlight'):
            self.plot_widget.removeItem(self.peak_highlight)
            del self.peak_highlight
            self.nearest_peak_idx = None

    def on_mouse_move(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = int(mouse_point.x())
            y = mouse_point.y()

            if 0 <= x < len(self.smoothed_data):
                self.floating_marker.setData([x], [self.smoothed_data[x]])
                if abs(y - self.smoothed_data[x]) < 10:
                    if x in self.x_points:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"X Point\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                    elif x in self.y_points:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"Y Point\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                    elif x in self.z_points:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"Z Point\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                    elif x in self.a_points:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"A Point\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                    elif x in self.b_points:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"B Point\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                    elif x in self.c_points:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"C Point\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                    else:
                        QToolTip.showText(self.plot_widget.mapToGlobal(pos.toPoint()), f"Signal\nIndex: {x}\nValue: {self.smoothed_data[x]:.4f}")
                else:
                    QToolTip.hideText()

                if QApplication.keyboardModifiers() == Qt.ControlModifier:
                    nearest_peak_idx = self.find_nearest_peak(x)
                    if nearest_peak_idx is not None:
                        self.highlight_peak(nearest_peak_idx)
                        self.nearest_peak_idx = nearest_peak_idx
                    else:
                        self.clear_peak_highlight()
                else:
                    self.clear_peak_highlight()
            else:
                self.floating_marker.setData([], [])
                QToolTip.hideText()
                self.clear_peak_highlight()

    def on_mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x = int(mouse_point.x())
            if 0 <= x < len(self.smoothed_data):
                if event.modifiers() == Qt.ControlModifier:  # 按住Ctrl鍵選擇第幾個peak到第幾個peak，中間的區段要標註穿戴狀態
                    if self.selected_start_peak_idx is None:
                        if self.nearest_peak_idx is not None:
                            self.selected_start_peak_idx = self.nearest_peak_idx
                        elif x == 0:  # 第一個點
                            self.selected_start_peak_idx = -1
                        elif x in self.x_points:
                            self.selected_start_peak_idx = self.x_points.index(x)
                    else:
                        if self.nearest_peak_idx is not None:
                            self.selected_end_peak_idx = self.nearest_peak_idx
                        elif x == len(self.smoothed_data) - 1:  # 最後一個點
                            self.selected_end_peak_idx = len(self.x_points)
                        elif x in self.x_points:
                            self.selected_end_peak_idx = self.x_points.index(x)
                        anomaly = self.label_anomaly()  # 調用標註異常區段的方法
                        if anomaly is not None:
                            (start_peak_idx, end_peak_idx, wearing_status) = anomaly
                            self.plot_anomaly_segment(start_peak_idx, end_peak_idx, wearing_status)
                            self.clear_segment(start_peak_idx, end_peak_idx, wearing_status)
                        self.selected_start_peak_idx = None
                        self.selected_end_peak_idx = None
                else:  # 普通點擊選擇點
                    if x not in self.selected_points:
                        self.selected_points.append(x)
                    else:
                        self.selected_points.remove(x)
                    self.update_selected_points_label()
                    self.plot_data()
        elif event.button() == Qt.RightButton:
            self.plot_widget.plotItem.vb.autoRange()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_X:
            self.label_selected_points('x')
        elif event.key() == Qt.Key_Y:
            self.label_selected_points('y')
        elif event.key() == Qt.Key_Z:
            self.label_selected_points('z')
        elif event.key() == Qt.Key_A:
            self.label_selected_points('a')
        elif event.key() == Qt.Key_B:
            self.label_selected_points('b')
        elif event.key() == Qt.Key_C:
            self.label_selected_points('c')
        elif event.key() == Qt.Key_Space:
            self.remove_selected_points()

    def label_selected_points(self, label):
        for point in self.selected_points:
            self.remove_selected_point(point)
            if label == 'x':
                self.x_points.append(point)
                self.x_points.sort()
            elif label == 'y':
                self.y_points.append(point)
                self.y_points.sort()
            elif label == 'z':
                self.z_points.append(point)
                self.z_points.sort()
            elif label == 'a':
                self.a_points.append(point)
                self.a_points.sort()
            elif label == 'b':
                self.b_points.append(point)
                self.b_points.sort()
            elif label == 'c':
                self.c_points.append(point)
                self.c_points.sort()
        self.selected_points.clear()
        self.update_selected_points_label()
        self.update_points_edit()
        self.plot_data()

    def remove_selected_point(self, point):
        if point is not None:
            if point in self.x_points:
                self.x_points.remove(point)
            elif point in self.y_points:
                self.y_points.remove(point)
            elif point in self.z_points:
                self.z_points.remove(point)
            elif point in self.a_points:
                self.a_points.remove(point)
            elif point in self.b_points:
                self.b_points.remove(point)
            elif point in self.c_points:
                self.c_points.remove(point)


    def remove_selected_points(self):
        for point in self.selected_points:
            self.remove_selected_point(point)
        self.selected_points.clear()
        self.update_selected_points_label()
        self.update_points_edit()
        self.plot_data()

    def update_selected_points_label(self):
        self.selected_points_label.setText(f"Selected Points: {self.selected_points}")


    def save_labeled_data(self):
        target_dir = os.path.join("labeled_DB", os.path.dirname(self.current_relative_path))
        os.makedirs(target_dir, exist_ok=True)  # 創建目標目錄
        target_path = os.path.join(target_dir, os.path.basename(self.current_relative_path))
        
        data = {
            "raw_data": self.raw_data,
            "smoothed_data": list(self.smoothed_data),
            "x_points": self.x_points,
            "y_points": self.y_points,
            "z_points": self.z_points,
            "a_points": self.a_points,
            "b_points": self.b_points,
            "c_points": self.c_points,
            "sample_rate": self.sample_rate,
            "anomaly_list": self.anomaly_list,
            "dataType": self.dataType,
            "macaddress": self.macaddress
            
            # 這裡可以添加更多需要保存的數據
        }
        
        def convert_np_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_np_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np_types(i) for i in obj]
            else:
                return obj
        
        converted_data = convert_np_types(data)
        
        # print('Saving labeled data:', data)
        with open(target_path, 'w') as f:
            json.dump(converted_data, f, indent=4)
        
        print("Saved labeled data to:", target_path)

    
    
    def update_file_tree(self, index):
        if index == 0:
            self.load_db_files(self.db_folder)
        else:
            self.load_db_files(self.labeled_db_folder)

    def load_db_files(self, db_folder = "DB"):
        
        self.file_tree_widget.clear()
        self.populate_tree(db_folder, self.file_tree_widget.invisibleRootItem())
        

    def populate_tree(self, folder, parent):
        #for item in os.listdir(folder):
        for item in natsorted(os.listdir(folder)):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path):
                folder_item = QTreeWidgetItem(parent)
                folder_item.setText(0, item)
                self.populate_tree(item_path, folder_item)
            elif item.endswith(".json"):
                file_item = QTreeWidgetItem(parent)
                file_item.setText(0, item)

    def load_selected_file(self, item, column):
        if item.childCount() == 0:  # 如果選擇的是檔案
            self.clear_data()
            file_path = self.get_file_path(item)
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if self.db_combo_box.currentIndex() == 0:  # DB
                self.current_relative_path = os.path.relpath(file_path, "DB")
                self.anomaly_list = []
                self.process_data(data)
            else:  # labeled_DB
                self.current_relative_path = os.path.relpath(file_path, "labeled_DB")
                self.load_labeled_data(data)
                self.reconstructed_signal = predict_reconstructed_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
                self.wear_corrected_signal = predict_corrected_reconstructed_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
            self.selected_points.clear()  # 清空Selected Point
            self.update_selected_points_label()  # 更新Selected Point的顯示
            self.update_points_edit()
            self.plot_data()

            self.window_start_slider.setValue(0)
            self.update_fft_plot()

    def update_fft_plot(self):

        self.window_start = self.window_start_slider.value()
        window_data = self.smoothed_data_100hz[self.window_start:self.window_start + self.window_size]
        # print(f'len(window_data): {len(window_data)}')
        if len(window_data) < self.window_size:
            return        
        self.fft_result = np.abs(np.fft.fft(window_data))#20 * np.log10(np.abs(np.fft.fft(window_data)))
        self.fft_freq = np.fft.fftfreq(len(window_data), 1/self.sample_rate)

        self.fft_plot_widget.clear()
        self.fft_plot_widget.plot(self.fft_freq[:len(self.fft_freq)//2], np.abs(self.fft_result[:len(self.fft_freq)//2]))
        self.fft_plot_widget.setXRange(0, 25)

        window_start_time = self.window_start / self.sample_rate
        self.window_start_label.setText(f"Window Start: {window_start_time:.2f}s")

    def load_labeled_data(self, data):
        self.raw_data = data['raw_data'] if 'raw_data' in data else []
        self.smoothed_data = data['smoothed_data'] if 'smoothed_data' in data else []
        #標準化
        self.standardized_data = (self.smoothed_data - np.mean(self.smoothed_data)) / np.std(self.smoothed_data)
        self.sample_rate = data['sample_rate']  if 'sample_rate' in data else 100
        self.smoothed_data_100hz = resample(self.smoothed_data, int(len(self.smoothed_data) * 100 // self.sample_rate))
        self.x_points = data['x_points'] if 'x_points' in data else []
        self.y_points = data['y_points'] if 'y_points' in data else []
        self.z_points = data['z_points'] if 'z_points' in data else []
        self.a_points = data['a_points'] if 'a_points' in data else []
        self.b_points = data['b_points'] if 'b_points' in data else []
        self.c_points = data['c_points'] if 'c_points' in data else []
        self.macaddress = data['macaddress'] if 'macaddress' in data else ""
        self.dataType = data['dataType'] if 'dataType' in data else "EPG"
        self.anomaly_list = data['anomaly_list'] if 'anomaly_list' in data else []

    def clear_anomaly_labels(self):
        self.anomaly_text_edit.clear()
        self.anomaly_list.clear()
        self.plot_data()

    def clear_data(self):
        self.raw_data = []
        self.smoothed_data = []
        self.standardized_data = []
        self.smoothed_data_100hz = []
        self.x_points = []
        self.y_points = []
        self.z_points = []
        self.a_points = []
        self.b_points = []
        self.c_points = []
        self.dataType = ""
        self.macaddress = ""
        
    def process_data(self, data):
        self.raw_data = [-value for packet in data['raw_data'] for value in packet['datas']]
        self.userID = data['user_id']
        self.sample_rate = data['sample_rate']
        self.dataType = data['dataType']
        self.macaddress = data['macaddress']
        print(f'Sample_rate: {self.sample_rate}')
        scale = int(3 * self.sample_rate / 100)
        self.smoothed_data = gaussian_smooth(self.raw_data, scale, scale/4)
        self.standardized_data = (self.smoothed_data - np.mean(self.smoothed_data)) / np.std(self.smoothed_data)
        if len(self.smoothed_data) > 0 : 
            self.smoothed_data_100hz = resample(self.smoothed_data, int(len(self.smoothed_data) * 100 // self.sample_rate))
            self.window_start_slider.setMaximum(len(self.smoothed_data_100hz) - self.window_size)
            self.window_start_slider.setValue(0)
            self.update_fft_plot()

            self.find_peaks_and_valleys(self.smoothed_data, int(self.sample_rate))#, 0.3, 0.9, 0.3, 0.03, 0.03, "on_peak", 0.06)
            self.update_latent_vector()
            # self.reconstructed_signal = predict_transformer_reconstructed_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
            self.reconstructed_signal = predict_reconstructed_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
            self.wear_corrected_signal = predict_corrected_reconstructed_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
            # self.reconstructed_signal = reconstruct_pulse_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
            # self.reconstructed_signal = predict_reconstructed_signal2(self.smoothed_data, int(self.sample_rate)) #predict_reconstructed_signal_pulse(self.smoothed_data, int(self.sample_rate), self.x_points)
            # self.reconstructed_signal = predict_LSTM_reconstructed_signal(self.smoothed_data, int(self.sample_rate), self.x_points)
        self.find_points()

    def update_latent_vector(self):
        self.encode_pulses()
        self.pulse_slider.setMaximum(len(self.latent_vectors) - 1)
        self.pulse_slider.setValue(0)
        self.update_latent_vector_display()        
    def encode_pulses(self):
        self.latent_vectors = []
        for i in range(len(self.x_points) - 1):
            start = self.x_points[i]
            end = self.x_points[i + 1]
            pulse = self.smoothed_data[start:end]
            
            # 插值到目标长度
            target_len = 100
            if len(pulse) > 1:
                interp_func = scipy.interpolate.interp1d(np.arange(len(pulse)), pulse, kind='linear')
                pulse_resampled = interp_func(np.linspace(0, len(pulse) - 1, target_len))
                pulse_tensor = torch.tensor(pulse_resampled, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    _, latent_vector = self.baseline_model3(pulse_tensor)
                
                self.latent_vectors.append(latent_vector.squeeze().cpu().numpy())

    def update_latent_vector_display(self):
        if self.latent_vectors:
            index = self.pulse_slider.value()
            pulse = self.get_current_pulse()
            if pulse is not None:
                amplitude_factor = self.amplitude_slider.value() / 100
                adjusted_pulse = pulse * amplitude_factor
                latent_vector = self.calculate_latent_vector(adjusted_pulse)
                
                # 更新 Pulse 索引标签
                self.pulse_index_label.setText(f"Pulse: {index}")
                
                # 计算时间
                start_time = self.x_points[index] / self.sample_rate
                end_time = self.x_points[index + 1] / self.sample_rate if index + 1 < len(self.x_points) else len(self.smoothed_data) / self.sample_rate
                time_diff = end_time - start_time
                
                # 格式化 latent vector 显示
                vector_str = " ".join(f"{v:.4f}" for v in latent_vector)
                display_text = f"Time length = {time_diff:.2f} ({start_time:.2f}s - {end_time:.2f}s)\n{vector_str}"
                
                self.latent_vector_display.setText(display_text)
            else:
                self.pulse_index_label.setText("Pulse: N/A")
                self.latent_vector_display.setText("No pulse available.")
        else:
            self.pulse_index_label.setText("Pulse: N/A")
            self.latent_vector_display.setText("No latent vectors available.")
                                                

    def get_file_path(self, item):
        path = [item.text(0)]
        while item.parent():
            item = item.parent()
            path.append(item.text(0))
        path.reverse()
        if self.db_combo_box.currentIndex() == 0:  # DB
            return os.path.join(self.db_folder, *path)
        else:  # labeled_DB
            return os.path.join(self.labeled_db_folder, *path)
        
    def find_peaks_and_valleys(self, data, sample_rate):#, drop_rate, drop_rate_gain, timer_init, timer_peak_refractory_period, peak_refinement_window, Vpp_method, Vpp_threshold):
        # peaks_top, peaks_bottom, envelope_plot_top, envelope_plot_bottom, Vpp_plot = find_peaks_helper(
        #     data, sample_rate, drop_rate, drop_rate_gain, timer_init, timer_peak_refractory_period, peak_refinement_window, Vpp_method, Vpp_threshold
        # )
        data = np.asarray(data, dtype=np.float32) 
        self.x_points = detect_peaks_from_signal(data, sample_rate)
        self.y_points = findValleysBasedOnPeaks(data, self.x_points, sample_rate)



    def find_points(self):
        results = find_epg_points(self.smoothed_data, self.x_points, self.y_points, self.sample_rate)
        # print(f'find_epg_points Results: {results}')
        for result in results:
            if len(result) == 2:
                z, a = result[0]
                b, c = result[1]
                if z != -1:
                    self.z_points.append(z)
                if a != -1:
                    self.a_points.append(a)
                if b != -1:
                    self.b_points.append(b)
                if c != -1:
                    self.c_points.append(c)



    def recalculate_y_points(self):
        if len(self.smoothed_data) > 0 and len(self.x_points) > 0:
            self.y_points = findValleysBasedOnPeaks(self.smoothed_data, self.x_points, self.sample_rate)
            self.update_points_edit()
            self.plot_data()
            self.clear_anomaly_segments()

    def recalculate_zabc_points(self):
        if len(self.smoothed_data) > 0 and len(self.x_points) > 0 and len(self.y_points) > 0:
            self.z_points = []
            self.a_points = []
            self.b_points = []
            self.c_points = []

            results = find_epg_points(self.smoothed_data, self.x_points, self.y_points, self.sample_rate)
            for result in results:
                if len(result) == 2:
                    z, a = result[0]
                    b, c = result[1]
                    if z != -1:
                        self.z_points.append(z)
                    if a != -1:
                        self.a_points.append(a)
                    if b != -1:
                        self.b_points.append(b)
                    if c != -1:
                        self.c_points.append(c)

            self.update_points_edit()
            self.plot_data()
            self.clear_anomaly_segments()



    def update_points_edit(self):
        self.x_points_edit.setText(', '.join(map(str, self.x_points)))
        self.y_points_edit.setText(', '.join(map(str, self.y_points)))
        self.z_points_edit.setText(', '.join(map(str, self.z_points)))
        self.a_points_edit.setText(', '.join(map(str, self.a_points)))
        self.b_points_edit.setText(', '.join(map(str, self.b_points)))
        self.c_points_edit.setText(', '.join(map(str, self.c_points)))
            
    def update_x_points_from_edit(self):
        try:
            self.x_points = [int(x) for x in self.x_points_edit.text().split(',') if x.strip()]
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')

    def update_y_points_from_edit(self):
        try:
            self.y_points = [int(y) for y in self.y_points_edit.text().split(',') if y.strip()]
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')

    def update_z_points_from_edit(self):
        try:
            self.z_points = [int(z) for z in self.z_points_edit.text().split(',') if z.strip()]
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')

    def update_a_points_from_edit(self):
        try:
            self.a_points = [int(a) for a in self.a_points_edit.text().split(',') if a.strip()]
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')

    def update_b_points_from_edit(self):
        try:
            self.b_points = [int(b) for b in self.b_points_edit.text().split(',') if b.strip()]
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')

    def update_c_points_from_edit(self):
        try:
            self.c_points = [int(c) for c in self.c_points_edit.text().split(',') if c.strip()]
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')

    def label_anomaly(self):
        if self.selected_start_peak_idx is not None and self.selected_end_peak_idx is not None:
            if self.selected_start_peak_idx <= self.selected_end_peak_idx:
                wearing_status, ok = QInputDialog.getItem(self, "Select Wearing Status", "Wearing Status:", ["underdamping", "overdamping", "noise", "moving", "not sure"], 0, False)
                if ok:
                    self.anomaly_list.append((self.selected_start_peak_idx, self.selected_end_peak_idx, wearing_status))
                    anomaly_label = f"Anomaly: Start Peak: {self.selected_start_peak_idx}, End Peak: {self.selected_end_peak_idx}, Wearing Status: {wearing_status}\n"
                    print(f'anomaly_label: {anomaly_label}')
                    self.anomaly_text_edit.append(anomaly_label)  # 將字符串附加到 QTextEdit
                    return (self.selected_start_peak_idx, self.selected_end_peak_idx, wearing_status)
            else:
                QMessageBox.warning(self, "Invalid Selection", "The start peak should come before the end peak.")
        else:
            QMessageBox.warning(self, "Incomplete Selection", "Please select both the start and end peaks.")


    def clear_segment(self, start_peak_idx, end_peak_idx, wearing_status):
        print(f'clear_segment: {start_peak_idx}, {end_peak_idx}, {wearing_status}')
        if wearing_status in ["moving", "noise"]:#移除所有start_peak_idx, end_peak_idx之間的xyzabc點
            start_x = self.x_points[start_peak_idx] if start_peak_idx != -1 else 0
            end_x = self.x_points[end_peak_idx] if end_peak_idx != len(self.x_points) else len(self.smoothed_data) - 1

            # self.x_points = [x for x in self.x_points if x < start_x or x > end_x]
            self.y_points = [y for y in self.y_points if y < start_x or y > end_x]
            self.z_points = [z for z in self.z_points if z < start_x or z > end_x]
            self.a_points = [a for a in self.a_points if a < start_x or a > end_x]
            self.b_points = [b for b in self.b_points if b < start_x or b > end_x]
            self.c_points = [c for c in self.c_points if c < start_x or c > end_x]

            self.update_points_edit()

    def clear_anomaly_segments(self):
        for anomaly in self.anomaly_list:
            start_peak_idx, end_peak_idx, wearing_status = anomaly
            self.clear_segment(start_peak_idx, end_peak_idx, wearing_status)            

    def plot_anomaly_segment(self, start_peak_idx, end_peak_idx, wearing_status):
        start_x = self.x_points[start_peak_idx] if start_peak_idx != -1 else 0
        end_x = self.x_points[end_peak_idx] if end_peak_idx != len(self.x_points) else len(self.smoothed_data) - 1
        print(f'plot_anomaly_segment: {start_x}, {end_x}, {wearing_status}')
        if wearing_status == "underdamping":
            color = (255, 0, 0)  # 紅色
        elif wearing_status == "overdamping": # 紫色
            color = (255, 0, 255)  # 紫色
        elif wearing_status == "noise": #深灰色
            color = (100, 100, 100)  # 灰色
        elif wearing_status == "moving": #深黃色
            color = (80, 100, 0)  # 黃色
        elif wearing_status == "not sure": #粉紅色
            color = (200, 100, 100)  # 粉紅色
        
        self.plot_widget.plot(list(range(start_x, end_x + 1)), self.smoothed_data[start_x:end_x + 1], pen=pg.mkPen(color=color, width=4), name=f'Anomaly ({wearing_status})')

    def add_noise_with_snr(self, data, target_snr_db):
        signal_power = np.mean(data ** 2)
        signal_power_db = 10 * np.log10(signal_power)

        noise_power_db = signal_power_db - target_snr_db
        noise_power = 10 ** (noise_power_db / 10)
        
        noise = np.sqrt(noise_power) * np.random.randn(len(data))
        
        noisy_data = data + noise
        return noisy_data
    def add_gaussian_noise_numpy(self, data):
        """
        Add Gaussian noise to the input data using NumPy.
        
        Args:
            data (np.ndarray): Input data.
            mean (float): Mean of the Gaussian distribution (default is 0).
            std (float): Standard deviation of the Gaussian distribution (default is 0.1).
            
        Returns:
            np.ndarray: Noisy data.
        """
        print(f'add_gaussian_noise_numpy: {data}')
        noise = np.random.normal(loc=0, scale=0.005, size=len(data))
        print(f'noise: {noise}')
        noisy_data = data + noise
        return noisy_data
    def resample_noise(self):
        if len(self.smoothed_data) > 0:
            self.noisy_data = self.add_gaussian_noise_numpy(self.smoothed_data)#self.add_noise_with_snr(self.smoothed_data, self.target_snr_db)
            self.plot_data()
        

    def plot_data(self):
        self.plot_widget.clear()
        symbolSize = 1
        if self.checkbox_raw_data.isChecked():
            self.plot_widget.plot(self.raw_data, pen=pg.mkPen(color=(200, 200, 200), width=1), name='Raw Data')

        if self.checkbox_smoothed_data.isChecked():
            self.plot_widget.plot(self.smoothed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Smoothed Data')

        if self.checkbox_reconstructed_signal.isChecked():
            self.plot_widget.plot(self.reconstructed_signal, pen=pg.mkPen(color=(100, 155, 255), width=2), name='Reconstructed Data')

        if self.checkbox_wear_corrected_signal.isChecked():
            self.plot_widget.plot(self.wear_corrected_signal, pen=pg.mkPen(color=(48, 80, 80), width=2), name='Wearing Corrected Data')

        if self.checkbox_noisy_data.isChecked() and len(self.noisy_data) > 0:
            self.plot_widget.plot(self.noisy_data, pen=pg.mkPen(color=(255, 0, 0), width=1), name='Noisy Data')


        # if self.latent_vectors:
        #     current_pulse = self.pulse_slider.value()
        #     if current_pulse < len(self.x_points) - 1:
        #         start = self.x_points[current_pulse]
        #         end = self.x_points[current_pulse + 1]
        #         x = np.arange(start, end)
        #         y = np.array(self.smoothed_data[start:end]).flatten()
                
        #         if len(x) == len(y):
        #             highlight = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color=(255, 255, 0, a  , width=3))
        #             self.plot_widget.addItem(highlight)
        #         else:
        #             print(f"Warning: x and y lengths do not match. x: {len(x)}, y: {len(y)}")


        for anomaly in self.anomaly_list: 
            start_peak_idx, end_peak_idx, wearing_status = anomaly
            self.plot_anomaly_segment(start_peak_idx, end_peak_idx, wearing_status)    

        if self.checkbox_x_points.isChecked():
            x_points_in_range = [i for i in self.x_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(x_points_in_range, [self.smoothed_data[i] for i in x_points_in_range], pen=None, symbol='o', symbolBrush=(0, 0, 0), symbolSize=15, name='X Points')

        if self.checkbox_y_points.isChecked():
            y_points_in_range = [i for i in self.y_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(y_points_in_range, [self.smoothed_data[i] for i in y_points_in_range], pen=None, symbol='o', symbolBrush=(0, 255, 255), symbolSize=15, name='Y Points')

        if self.checkbox_z_points.isChecked():
            z_points_in_range = [i for i in self.z_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(z_points_in_range, [self.smoothed_data[i] for i in z_points_in_range], pen=None, symbol='o', symbolBrush=(255, 0, 255), symbolSize=15, name='Z Points')

        if self.checkbox_a_points.isChecked():
            a_points_in_range = [i for i in self.a_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(a_points_in_range, [self.smoothed_data[i] for i in a_points_in_range], pen=None, symbol='o', symbolBrush=(128, 0, 128), symbolSize=15, name='A Points')

        if self.checkbox_b_points.isChecked():
            b_points_in_range = [i for i in self.b_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(b_points_in_range, [self.smoothed_data[i] for i in b_points_in_range], pen=None, symbol='o', symbolBrush=(128, 128, 0), symbolSize=15, name='B Points')

        if self.checkbox_c_points.isChecked():
            c_points_in_range = [i for i in self.c_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(c_points_in_range, [self.smoothed_data[i] for i in c_points_in_range], pen=None, symbol='o', symbolBrush=(0, 128, 128), symbolSize=15, name='C Points')

        self.legend.clear()
        for item in self.plot_widget.listDataItems():
            if isinstance(item, pg.PlotDataItem):
                self.legend.addItem(item, item.name())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())