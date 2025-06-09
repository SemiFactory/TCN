import sys
import pandas as pd
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
)

# ===== TCN 분류기 정의 =====
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x + res

class TCN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TCN, self).__init__()
        layers = []
        channels = [input_channels, 32, 64, 64]
        for i in range(len(channels) - 1):
            layers.append(ResidualBlock(channels[i], channels[i + 1],
                                        kernel_size=7, dilation=1, dropout=0.101497))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# ===== PyQt5 GUI =====
class StateClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("예측 센서값 상태 분류기")
        self.model = None
        self.model_path = None
        self.file_path = None
        self.data_tensor = None
        self.index = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("CSV 파일과 분류 모델을 선택하세요.")
        layout.addWidget(self.label)

        self.model_btn = QPushButton("모델 선택 (.pth)")
        self.model_btn.clicked.connect(self.select_model)
        layout.addWidget(self.model_btn)

        self.csv_btn = QPushButton("센서 예측값 CSV 업로드")
        self.csv_btn.clicked.connect(self.upload_csv)
        layout.addWidget(self.csv_btn)

        self.predict_btn = QPushButton("예측 순차 실행")
        self.predict_btn.clicked.connect(self.predict_next)
        layout.addWidget(self.predict_btn)

        self.output_box = QTextEdit()
        layout.addWidget(self.output_box)

        self.setLayout(layout)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델 선택", "", "Model Files (*.pth)")
        if path:
            self.model_path = path
            self.label.setText(f"모델 파일 선택됨: {path}")
            self.try_load_model()

    def upload_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV 선택", "", "CSV Files (*.csv)")
        if path:
            self.file_path = path
            self.label.setText(f"CSV 업로드 완료: {path}")
            self.load_csv_data()

    def load_csv_data(self):
        try:
            df = pd.read_csv(self.file_path)
            feature_cols = [f"Pred_Sensor{i+1}" for i in range(7)]
            if not all(col in df.columns for col in feature_cols):
                self.output_box.setText("❌ CSV 파일에 필요한 센서 컬럼이 없습니다.")
                return
            self.data_tensor = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            self.index = 0
            self.output_box.setText("✅ CSV 로딩 성공. 예측을 실행하세요.")
        except Exception as e:
            self.output_box.setText(f"❌ CSV 로드 실패: {str(e)}")

    def try_load_model(self):
        try:
            self.model = TCN(input_channels=7, num_classes=4)
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.output_box.setText("✅ 모델 로드 성공!")
        except Exception as e:
            self.output_box.setText(f"❌ 모델 로드 실패: {str(e)}")

    def predict_next(self):
        if self.model is None or self.data_tensor is None:
            self.output_box.setText("❗ 모델과 데이터를 먼저 불러오세요.")
            return
        if self.index >= len(self.data_tensor):
            self.output_box.setText("✅ 모든 샘플 예측 완료.")
            return

        input_tensor = self.data_tensor[self.index].unsqueeze(0).unsqueeze(1)  # [1, 1, 7]
        input_tensor = input_tensor.transpose(1, 2)  # [1, 7, 1]

        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        self.output_box.append(f"[{self.index + 1}] 예측된 상태: {pred}")
        self.index += 1

# ===== 앱 실행 =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = StateClassifierGUI()
    gui.show()
    sys.exit(app.exec_())
