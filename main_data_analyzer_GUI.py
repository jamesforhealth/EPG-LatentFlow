import sys
import os
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem, QSplitter, QCheckBox, QLineEdit, QTextEdit, QToolTip
from PyQt5.QtGui import QMouseEvent, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
import pyqtgraph as pg
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from influxDB_downloader import get_user_sessions, get_session_data, format_timestamp, sanitize_filename

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

    for peak in peaks:
        foundValley = False

        for i in range(1, searchRange + 1):
            if peak + i + 1 >= len(data):
                break
            if data[peak + i] < data[peak + i - 1] and data[peak + i] < data[peak + i + 1]:
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
            else: #找到的那一組是bc點
                found_points = [(-1, -1), found_points[0]]
        elif found_points[0][0] - peak_idx > systolica_range: #找到不少於兩組但第一組是bc點
            found_points = [(-1, -1), found_points[0]]
        
        elif found_points[1][0] - peak_idx <= systolica_range: #找到不少於兩組但第二組還不是bc點
            found_points = [found_points[0], found_points[2]] if len(found_points) > 2 else [found_points[0], (-1, -1)]  #如果za附近多於一組則會出現例外，但不好處理    


        results.append(found_points)

    return results

def gaussian_smooth(input, window_size, sigma):
    half_window = window_size // 2
    output = np.zeros_like(input)
    weights = np.zeros(window_size)
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
        self.setWindowTitle("Waveform Labeling")
        self.resize(1200, 800)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=True, y=False)
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
        self.x_points = []
        self.y_points = []
        self.z_points = []
        self.a_points = []
        self.b_points = []
        self.c_points = []
        self.sample_rate = 100
        self.selected_points = []

        self.selected_points_label = QLabel("Selected Points: []")

        self.file_tree_widget = QTreeWidget()
        self.file_tree_widget.setHeaderLabels(["DB Files"])
        self.file_tree_widget.itemClicked.connect(self.load_selected_file)

        self.user_id_edit = QLineEdit()
        self.user_id_edit.setPlaceholderText("Enter User ID")

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_user_data)

        user_id_layout = QHBoxLayout()
        user_id_layout.addWidget(QLabel("User ID:"))
        user_id_layout.addWidget(self.user_id_edit)
        user_id_layout.addWidget(self.reload_button)

        file_tree_layout = QVBoxLayout()
        file_tree_layout.addLayout(user_id_layout)
        file_tree_layout.addWidget(self.file_tree_widget)

        self.checkbox_raw_data = QCheckBox("Raw Data")
        self.checkbox_raw_data.setChecked(True)
        self.checkbox_raw_data.stateChanged.connect(self.plot_data)

        self.checkbox_smoothed_data = QCheckBox("Smoothed Data")
        self.checkbox_smoothed_data.setChecked(True)
        self.checkbox_smoothed_data.stateChanged.connect(self.plot_data)

        self.checkbox_x_points = QCheckBox("X Points")
        self.checkbox_x_points.setChecked(True)
        self.checkbox_x_points.stateChanged.connect(self.plot_data)

        self.checkbox_y_points = QCheckBox("Y Points")
        self.checkbox_y_points.setChecked(True)
        self.checkbox_y_points.stateChanged.connect(self.plot_data)

        self.checkbox_z_points = QCheckBox("Z Points")
        self.checkbox_z_points.setChecked(True)
        self.checkbox_z_points.stateChanged.connect(self.plot_data)

        self.checkbox_a_points = QCheckBox("A Points")
        self.checkbox_a_points.setChecked(True)
        self.checkbox_a_points.stateChanged.connect(self.plot_data)

        self.checkbox_b_points = QCheckBox("B Points")
        self.checkbox_b_points.setChecked(True)
        self.checkbox_b_points.stateChanged.connect(self.plot_data)

        self.checkbox_c_points = QCheckBox("C Points")
        self.checkbox_c_points.setChecked(True)
        self.checkbox_c_points.stateChanged.connect(self.plot_data)

        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.checkbox_raw_data)
        checkbox_layout.addWidget(self.checkbox_smoothed_data)
        checkbox_layout.addWidget(self.checkbox_x_points)
        checkbox_layout.addWidget(self.checkbox_y_points)
        checkbox_layout.addWidget(self.checkbox_z_points)
        checkbox_layout.addWidget(self.checkbox_a_points)
        checkbox_layout.addWidget(self.checkbox_b_points)
        checkbox_layout.addWidget(self.checkbox_c_points)

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


        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addLayout(checkbox_layout)

        main_layout = QSplitter(Qt.Vertical)
        main_layout.addWidget(QWidget())
        main_layout.addWidget(QWidget())

        main_layout.widget(0).setLayout(plot_layout)
        main_layout.widget(1).setLayout(points_layout)

        main_layout.setStretchFactor(0, 3)
        main_layout.setStretchFactor(1, 1)

        central_widget = QSplitter(Qt.Horizontal)
        central_widget.addWidget(QWidget())
        central_widget.addWidget(QWidget())

        central_widget.widget(0).setLayout(file_tree_layout)
        central_widget.widget(1).setLayout(QVBoxLayout())
        central_widget.widget(1).layout().addWidget(main_layout)

        central_widget.setStretchFactor(0, 1)
        central_widget.setStretchFactor(1, 3)

        self.setCentralWidget(central_widget)

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)

        self.load_db_files()

        self.floating_marker = pg.ScatterPlotItem(size=10, pen=pg.mkPen(color=(255, 0, 0), width=2), brush=pg.mkBrush(255, 255, 255, 120))
        self.plot_widget.addItem(self.floating_marker)

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

    def on_mouse_move(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = int(mouse_point.x())
            y = mouse_point.y()

            if 0 <= x < len(self.smoothed_data):
                self.floating_marker.setData([x], [self.smoothed_data[x]])
                if  abs(y - self.smoothed_data[x]) < 10:
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
            else:
                self.floating_marker.setData([], [])
                QToolTip.hideText()

    def on_mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x = int(mouse_point.x())
            if 0 <= x < len(self.smoothed_data):
                if x not in self.selected_points:
                    self.selected_points.append(x)
                else:
                    self.selected_points.remove(x)
                self.update_selected_points_label()
                self.plot_data()
        elif event.button() == Qt.RightButton:
            self.plot_widget.plotItem.vb.autoRange()

    def on_mouse_double_clicked(self, event):
        if event.button() == Qt.LeftButton:
            self.selected_points.clear()
            self.update_selected_points_label()
            self.plot_data()

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
        target_dir = os.path.join("point_labelled_DB", os.path.dirname(self.current_relative_path))
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
            # 這裡可以添加更多需要保存的數據
        }
        
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("Saved labeled data to:", target_path)
        
    def load_db_files(self):
        db_folder = "DB"
        self.file_tree_widget.clear()
        self.populate_tree(db_folder, self.file_tree_widget.invisibleRootItem())
        

    def populate_tree(self, folder, parent):
        for item in os.listdir(folder):
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
            file_path = self.get_file_path(item)
            self.current_relative_path = os.path.relpath(file_path, "DB")
            labeled_file_path = os.path.join("point_labelled_DB", self.current_relative_path)
            if os.path.exists(labeled_file_path):
                with open(labeled_file_path, 'r') as file:
                    labeled_data = json.load(file)
                self.clear_data()
                self.load_labeled_data(labeled_data)
            else:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                self.clear_data()
                self.process_data(data)

            self.selected_points.clear()  # 清空Selected Point
            self.update_selected_points_label()  # 更新Selected Point的顯示
            self.update_points_edit()
            self.plot_data()
    def load_labeled_data(self, data):
        self.raw_data = data['raw_data']
        self.smoothed_data = data['smoothed_data']
        self.x_points = data['x_points']
        self.y_points = data['y_points']
        self.z_points = data['z_points']
        self.a_points = data['a_points']
        self.b_points = data['b_points']
        self.c_points = data['c_points']            
    def clear_data(self):
        self.raw_data = []
        self.smoothed_data = []
        self.x_points = []
        self.y_points = []
        self.z_points = []
        self.a_points = []
        self.b_points = []
        self.c_points = []
    def process_data(self, data):
        self.raw_data = [-value for packet in data['raw_data'] for value in packet['datas']]
        self.userID = data['user_id']
        self.sample_rate = data['sample_rate']
        print(f'Sample_rate: {self.sample_rate}')
        scale = int(3 * self.sample_rate / 100)
        self.smoothed_data = gaussian_smooth(self.raw_data, scale, scale/4)
        if len(self.smoothed_data) > 0 : self.find_peaks_and_valleys(self.smoothed_data, self.sample_rate, 0.3, 0.9, 0.3, 0.03, 0.03, "on_peak", 0.06)
        self.find_points()


    def get_file_path(self, item):
        path = [item.text(0)]
        while item.parent():
            item = item.parent()
            path.append(item.text(0))
        path.reverse()
        return os.path.join("DB", *path)

    def find_peaks_and_valleys(self, data, sample_rate, drop_rate, drop_rate_gain, timer_init, timer_peak_refractory_period, peak_refinement_window, Vpp_method, Vpp_threshold):
        peaks_top, peaks_bottom, envelope_plot_top, envelope_plot_bottom, Vpp_plot = find_peaks_helper(
            data, sample_rate, drop_rate, drop_rate_gain, timer_init, timer_peak_refractory_period, peak_refinement_window, Vpp_method, Vpp_threshold
        )
        self.x_points = peaks_top
        self.y_points = findValleysBasedOnPeaks(data, peaks_top, sample_rate)

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





    def update_points_edit(self):
        self.x_points_edit.setText(', '.join(map(str, self.x_points)))
        self.y_points_edit.setText(', '.join(map(str, self.y_points)))
        self.z_points_edit.setText(', '.join(map(str, self.z_points)))
        self.a_points_edit.setText(', '.join(map(str, self.a_points)))
        self.b_points_edit.setText(', '.join(map(str, self.b_points)))
        self.c_points_edit.setText(', '.join(map(str, self.c_points)))
            
    def update_x_points_from_edit(self):
        try:
            self.x_points = list(map(int, self.x_points_edit.text().split(',')))
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')
    def update_y_points_from_edit(self):
        try:
            self.y_points = list(map(int, self.y_points_edit.text().split(',')))
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')  
    def update_z_points_from_edit(self):
        try:
            self.z_points = list(map(int, self.z_points_edit.text().split(',')))
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')  
    def update_a_points_from_edit(self):
        try:
            self.a_points = list(map(int, self.a_points_edit.text().split(',')))
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')  
    def update_b_points_from_edit(self):
        try:
            self.b_points = list(map(int, self.b_points_edit.text().split(',')))
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')  
    def update_c_points_from_edit(self):
        try:
            self.c_points = list(map(int, self.c_points_edit.text().split(',')))
            self.plot_data()
        except ValueError as e:
            print(f'ValueError: {e}')          

    def plot_data(self):
        self.plot_widget.clear()

        if self.checkbox_raw_data.isChecked():
            self.plot_widget.plot(self.raw_data, pen=pg.mkPen(color=(200, 200, 200), width=1), name='Raw Data')

        if self.checkbox_smoothed_data.isChecked():
            self.plot_widget.plot(self.smoothed_data, pen=pg.mkPen(color=(0, 0, 255), width=2), name='Smoothed Data')

        if self.checkbox_x_points.isChecked():
            x_points_in_range = [i for i in self.x_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(x_points_in_range, [self.smoothed_data[i] for i in x_points_in_range], pen=None, symbol='o', symbolBrush=(255, 255, 0), symbolSize=7, name='X Points')

        if self.checkbox_y_points.isChecked():
            y_points_in_range = [i for i in self.y_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(y_points_in_range, [self.smoothed_data[i] for i in y_points_in_range], pen=None, symbol='o', symbolBrush=(0, 255, 255), symbolSize=7, name='Y Points')

        if self.checkbox_z_points.isChecked():
            z_points_in_range = [i for i in self.z_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(z_points_in_range, [self.smoothed_data[i] for i in z_points_in_range], pen=None, symbol='o', symbolBrush=(255, 0, 255), symbolSize=7, name='Z Points')

        if self.checkbox_a_points.isChecked():
            a_points_in_range = [i for i in self.a_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(a_points_in_range, [self.smoothed_data[i] for i in a_points_in_range], pen=None, symbol='o', symbolBrush=(128, 0, 128), symbolSize=7, name='A Points')

        if self.checkbox_b_points.isChecked():
            b_points_in_range = [i for i in self.b_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(b_points_in_range, [self.smoothed_data[i] for i in b_points_in_range], pen=None, symbol='o', symbolBrush=(128, 128, 0), symbolSize=7, name='B Points')

        if self.checkbox_c_points.isChecked():
            c_points_in_range = [i for i in self.c_points if 0 <= i < len(self.smoothed_data)]
            self.plot_widget.plot(c_points_in_range, [self.smoothed_data[i] for i in c_points_in_range], pen=None, symbol='o', symbolBrush=(0, 128, 128), symbolSize=7, name='C Points')

        self.legend.clear()
        for item in self.plot_widget.listDataItems():
            if isinstance(item, pg.PlotDataItem):
                self.legend.addItem(item, item.name())
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())