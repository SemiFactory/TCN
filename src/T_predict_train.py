import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import TCN

# ==========================================
# 하이퍼파라미터 설정
# ==========================================
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
KERNEL_SIZE = 7
NUM_CHANNELS = [32, 64, 64, 64]

PATIENCE = 3
MIN_DELTA = 1e-4  # 개선 간주 최소 변화량

# ==========================================
# 데이터 로드
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "..", "results")

X_path = os.path.join(results_dir, "X_windows_predict.npy")
y_path = os.path.join(results_dir, "y_states_pridict.npy")

if not os.path.exists(X_path) or not os.path.exists(y_path):
    raise FileNotFoundError("[❌] 전처리된 데이터가 없습니다. 먼저 T_state_prediction_s_window.py 를 실행하세요.")

X = np.load(X_path)
y = np.load(y_path)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

n_samples, window_size, n_features = X.shape

print(f"[ℹ️] Samples: {n_samples}, Window Size: {window_size}, Features: {n_features}")

# ==========================================
# 모델, 손실함수, 옵티마이저 정의
# ==========================================
SENSOR_DIM = y.shape[1]
model = TCN(input_size=n_features, output_size=SENSOR_DIM, num_channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE)
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 학습 루프 + 조기 종료
# ==========================================
model.train()
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    permutation = torch.randperm(n_samples)
    total_loss = 0.0
    total_mae = 0.0

    for i in range(0, n_samples, BATCH_SIZE):
        indices = permutation[i:i + BATCH_SIZE]
        batch_x = X[indices]
        batch_y = y[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion_mse(outputs, batch_y)
        mae = criterion_mae(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae.item()

    avg_loss = total_loss / (n_samples / BATCH_SIZE)
    avg_mae = total_mae / (n_samples / BATCH_SIZE)
    print(f"[Epoch {epoch + 1}/{EPOCHS}] Train MSE: {avg_loss:.4f} | MAE: {avg_mae:.4f}")

    # 조기 종료 체크
    if best_loss - avg_loss > MIN_DELTA:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[⏹️] 조기 종료: {epoch + 1} 에포크에서 개선 없음")
            break

# ==========================================
# 모델 저장
# ==========================================
model_save_dir = os.path.join(results_dir, "models")
os.makedirs(model_save_dir, exist_ok=True)

model_save_path = os.path.join(model_save_dir, "tcn_state_predictor.pth")
torch.save(model.state_dict(), model_save_path)

print(f"[✅] 모델 저장 완료: {model_save_path}") 