import sys
import torch
import numpy as np
import scipy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from model_pulse_representation import EPGBaselinePulseAutoencoder


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=4, dpi=100):  # 調整寬度為10
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self, model, mean_values, std_values, target_len=100):
        super().__init__()

        self.model = model
        self.mean_values = mean_values
        self.std_values = std_values
        self.target_len = target_len
        self.latent_vector = mean_values.copy()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Pulse Autoencoder Latent Space Explorer')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        self.canvas = MplCanvas(self, width=10, height=4, dpi=100)
        layout.addWidget(self.canvas)

        sliders_layout = QVBoxLayout()
        self.sliders = []
        self.slider_labels = []

        for i in range(30):
            slider_layout = QHBoxLayout()

            label = QLabel(f'Dimension {i+1}')
            slider_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-400)
            slider.setMaximum(400)
            slider.setValue(0)
            slider.setTickInterval(100)
            slider.valueChanged.connect(self.update_plot)
            slider_layout.addWidget(slider)

            slider_value_label = QLabel("0.0")
            slider_layout.addWidget(slider_value_label)

            sliders_layout.addLayout(slider_layout)
            self.sliders.append(slider)
            self.slider_labels.append(slider_value_label)

        layout.addLayout(sliders_layout)

        button_layout = QVBoxLayout()

        # 設定一個reset按鈕把每個維度的標準差刻度都變成0回到初始狀態
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_sliders)
        button_layout.addWidget(reset_button)

        # 設定一個sample按鈕隨機產生新的值
        sample_button = QPushButton("Sample")
        sample_button.clicked.connect(self.sample_latent_vector)
        button_layout.addWidget(sample_button)

        layout.addLayout(button_layout)
                
        self.update_plot()

    def reset_sliders(self):
        for slider in self.sliders:
            slider.setValue(0)
        self.update_plot()

    def sample_latent_vector(self):
        for i in range(30):
            random_value = np.random.normal(self.mean_values[i], self.std_values[i])
            self.latent_vector[i] = random_value
            std_devs = (random_value - self.mean_values[i]) / self.std_values[i]
            self.sliders[i].setValue(int(std_devs * 100))
        self.update_plot()

    def update_plot(self):
        for i, slider in enumerate(self.sliders):
            value = slider.value() / 100.0
            self.latent_vector[i] = self.mean_values[i] + value * self.std_values[i]
            self.slider_labels[i].setText(f"{value:.1f}")

        latent_tensor = torch.tensor(self.latent_vector, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            decoded_signal = self.model.dec(latent_tensor)
        decoded_signal = decoded_signal.squeeze().cpu().numpy()

        self.canvas.axes.clear()
        self.canvas.axes.plot(decoded_signal)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'pulse_interpolate_autoencoder.pth'
    model = EPGBaselinePulseAutoencoder(target_len=100).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mean_values = np.array([ 0.42873028,  0.00295145,  0.01233245,  0.01094226,  0.0499148 , -0.0695532 ,
 -0.82059063, -0.49606685, -1.16227683,  0.03385485, -0.02914557,  0.00937726,
  0.03650687,  0.05714999,  0.10468714, -1.9215688 ,  1.46757363, -0.00462542,
  0.03864986, -0.98842405, -0.12197125,  0.01546735,  0.64747278,  0.64712652,
 -1.72601604, -0.02611889, -1.00351373, -1.00867921,  0.25800518,  0.69864992])
    std_values = np.array([0.27821073, 0.33010676, 0.67849293, 0.36255858, 0.27987218, 0.53718241,
 0.37071564, 0.76043965, 0.39918327, 0.27799054, 0.38550461, 0.47111382,
 0.41424568, 0.49426754, 0.32321763, 0.78529666, 0.9122345 , 0.20872541,
 0.52895383, 0.34134531, 0.37625366, 0.2697667 , 0.44102414, 0.83986459,
 0.62443896, 0.43329468, 0.71877118, 0.51694429, 0.89549531, 0.39520511])

    window = MainWindow(model, mean_values, std_values)
    window.show()
    sys.exit(app.exec_())
