import sys
import pandas as pd
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from pyvistaqt import BackgroundPlotter
import random

class MainWindow(QMainWindow):
    def __init__(self, csv_path):
        super().__init__()
        self.setWindowTitle("PyVista in PyQt - Random Colors for Labels")
        self.resize(800, 600)

        # 創建主窗口的中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 使用垂直佈局
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # 初始化 PyVista Plotter
        self.plotter = BackgroundPlotter(show=False)
        layout.addWidget(self.plotter.app_window)

        # 從 CSV 檔案讀取數據
        data = self.load_csv_data(csv_path)

        # 添加 3D 點雲
        self.add_pyvista_content(data)

    def load_csv_data(self, csv_path):
        """
        從 CSV 檔案讀取數據，並返回 DataFrame。
        假設 CSV 的結構為：
        X, Y, Z, (忽略列), Label
        """
        try:
            df = pd.read_csv(csv_path, header=None)
            return df
        except Exception as e:
            print(f"無法讀取 CSV 文件: {e}")
            sys.exit(1)

    def generate_unique_colors(self, num_colors):
        """
        生成不重複的隨機 RGB 顏色。
        """
        colors = set()
        while len(colors) < num_colors:
            # 隨機生成 RGB 顏色，每個值範圍 0-1
            colors.add(tuple(np.random.rand(3)))
        return list(colors)

    def add_pyvista_content(self, data):
        # 提取 XYZ 和 Label
        x, y, z = data[0], data[1], data[2]
        labels = data[4]

        # 唯一標籤分組，生成隨機顏色
        unique_labels = labels.unique()
        colors = self.generate_unique_colors(len(unique_labels))
        color_map = {label: color for label, color in zip(unique_labels, colors)}

        # 分配顏色到每個點
        point_colors = np.array([color_map[label] for label in labels])

        # 將數據組合為 PyVista 的 PolyData
        points = np.column_stack((x, y, z))
        point_cloud = pv.PolyData(points)

        # 添加 RGB 顏色屬性
        point_cloud['RGB'] = point_colors

        # 添加到 PyVista Plotter
        self.plotter.add_mesh(
            point_cloud,
            scalars='RGB',
            rgb=True,  # 啟用 RGB 顏色
            point_size=10,
            render_points_as_spheres=True,
        )
        # 创建坐标轴线 (黑色)
        origin = [0, 0, 0]
        x_line = np.array([[0, 0, 0], [1, 0, 0]]) * max(x.max(), y.max(), z.max())  # X轴
        y_line = np.array([[0, 0, 0], [0, 1, 0]]) * max(x.max(), y.max(), z.max())  # Y轴
        z_line = np.array([[0, 0, 0], [0, 0, 1]]) * max(x.max(), y.max(), z.max())  # Z轴

        # 将线段添加到渲染器中
        self.plotter.add_lines(x_line, color="black", width=2, label="X-axis")
        self.plotter.add_lines(y_line, color="black", width=2, label="Y-axis")
        self.plotter.add_lines(z_line, color="black", width=2, label="Z-axis")
        
# 啟動 PyQt5 應用
if __name__ == "__main__":
    # 替換為您的 CSV 檔案路徑
    csv_file_path = "U_large_name_40.csv"

    app = QApplication(sys.argv)
    window = MainWindow(csv_file_path)
    window.show()
    sys.exit(app.exec_())
