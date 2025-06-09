import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os # 그래프 누적되서 나오는거
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------------------------------------
# [1] 경로 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data", "sensor_predictions_only.csv")
model_path = os.path.join(project_root, "results", "logistic_state_classifier.pkl")

# --------------------------------------------------
# [2] 데이터 및 모델 로딩
# --------------------------------------------------
df = pd.read_csv(data_path)
model = joblib.load(model_path)

# sensor_cols: CSV에 있는 7개 센서 컬럼만 추출
sensor_cols = list(df.columns)

# --------------------------------------------------
# [3] GUI 초기화
# --------------------------------------------------
root = tk.Tk()
root.title("🔬 실시간 OHT 실험실 대시보드")
root.geometry("1800x1000")
style = ttk.Style()
style.theme_use("default")
style.configure("Treeview", rowheight=25)

# --------------------------------------------------
# [4] PanedWindow를 사용해 상/하단을 1:1로 분할
# --------------------------------------------------
paned = ttk.PanedWindow(root, orient=tk.VERTICAL)
paned.pack(fill=tk.BOTH, expand=True)

# 상단 프레임 (그래프)
top_frame = ttk.Frame(paned)
paned.add(top_frame, weight=1)   # 상단에 비중(weight=1)

# 하단 프레임 (테이블 + State 그래프)
bottom_frame = ttk.Frame(paned)
paned.add(bottom_frame, weight=1)  # 하단에 비중(weight=1)

# --------------------------------------------------
# [5] 상단 영역: 1×7 서브플롯
# --------------------------------------------------
graph_frame = ttk.LabelFrame(top_frame, text="📉 센서별 실시간 변화 (7분할)")
graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# 1행×7열 서브플롯 생성
fig_sensor, axs_sensor = plt.subplots(nrows=1, ncols=7, figsize=(20, 3), sharey=False)
fig_sensor.tight_layout(pad=3.0)

for i, col in enumerate(sensor_cols):
    axs_sensor[i].set_title(col)
    axs_sensor[i].set_xlabel("샘플")
    axs_sensor[i].set_ylabel("값")
    axs_sensor[i].grid(True)

canvas_sensor = FigureCanvasTkAgg(fig_sensor, master=graph_frame)
canvas_sensor.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 센서용 데이터 구조
sensor_data = {col: [] for col in sensor_cols}
sensor_index = []

def update_sensor_graph():
    """각 서브플롯(i)에 해당 센서값을 갱신."""
    for ax in axs_sensor:
        ax.clear()

    for i, col in enumerate(sensor_cols):
        axs_sensor[i].plot(sensor_index, sensor_data[col], color='tab:blue')
        axs_sensor[i].set_title(col)
        axs_sensor[i].set_xlabel("샘플")
        axs_sensor[i].set_ylabel("값")
        axs_sensor[i].grid(True)

    fig_sensor.tight_layout(pad=3.0)
    canvas_sensor.draw()

# --------------------------------------------------
# [6] 하단 영역 구성 (좌: 테이블, 우: State 꺾은선)
# --------------------------------------------------
# 6-1) 좌측 센서 테이블
table_frame = ttk.LabelFrame(bottom_frame, text="📋 센서 상태 테이블")
table_frame.pack(side="left", fill=tk.BOTH, expand=True, padx=5, pady=5)

tree = ttk.Treeview(table_frame, show="headings")
columns = sensor_cols + ["Predicted State"]
tree["columns"] = columns
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor="center")
tree.pack(side="left", fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y")

tree.tag_configure("state_0", background="#c8facc")  # 정상
tree.tag_configure("state_1", background="#fff5b1")  # 주의
tree.tag_configure("state_2", background="#ffcb9a")  # 경고
tree.tag_configure("state_3", background="#ff8a8a")  # 위험

# 6-2) 우측 State 꺾은선 그래프
state_frame = ttk.LabelFrame(bottom_frame, text="📊 상태(State) 추이")
state_frame.pack(side="right", fill=tk.BOTH, expand=True, padx=5, pady=5)

fig_state, ax_state = plt.subplots(figsize=(6, 4))
canvas_state = FigureCanvasTkAgg(fig_state, master=state_frame)
canvas_state.get_tk_widget().pack(fill=tk.BOTH, expand=True)

state_history = []

def update_state_graph():
    """state_history 값을 꺾은선 그래프로 갱신."""
    ax_state.clear()
    ax_state.plot(range(len(state_history)), state_history, marker='o', linestyle='-', color='red')
    ax_state.set_title("State 값 변화")
    ax_state.set_xlabel("샘플")
    ax_state.set_ylabel("State")
    ax_state.set_yticks([0, 1, 2, 3])
    ax_state.grid(True)
    fig_state.tight_layout(pad=3.0)
    canvas_state.draw()

# --------------------------------------------------
# [7] 제어 버튼
# --------------------------------------------------
control_frame = ttk.Frame(bottom_frame)
control_frame.pack(side="bottom", pady=5)

start_btn = ttk.Button(control_frame, text="▶ 실시간 탐색 시작")
stop_btn = ttk.Button(control_frame, text="⏹ 탐색 중단", state="disabled")
start_btn.pack(side="left", padx=10)
stop_btn.pack(side="left", padx=10)

# --------------------------------------------------
# [8] 시뮬레이션 로직
# --------------------------------------------------
stop_simulation = False

def simulate():
    global stop_simulation
    for idx, row in df.iterrows():
        if stop_simulation:
            break

        # (1) 예측을 위해 DataFrame 생성
        features = pd.DataFrame([row.values], columns=sensor_cols)
        pred = model.predict(features)[0]  # 0,1,2,3

        # (2) 센서 그래프 데이터 추가
        sensor_index.append(idx)
        for col in sensor_cols:
            sensor_data[col].append(row[col])
        update_sensor_graph()

        # (3) State 그래프 데이터 추가
        state_history.append(pred)
        update_state_graph()

        # (4) 테이블에 새 행 추가
        try:
            values = list(row.values) + [pred]
            tag = f"state_{pred}"
            tree.insert("", tk.END, values=values, tags=(tag,))
            tree.see(tree.get_children()[-1])
        except tk.TclError:
            # GUI가 닫혔거나 유효하지 않으면 루프 탈출
            break

        time.sleep(1)

    # 시뮬레이션 종료 뒤 버튼 복원
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")

def start():
    global stop_simulation
    stop_simulation = False
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")
    threading.Thread(target=simulate, daemon=True).start()

def stop():
    global stop_simulation
    stop_simulation = True

start_btn.config(command=start)
stop_btn.config(command=stop)

# --------------------------------------------------
# [9] GUI 메인루프 실행
# --------------------------------------------------
root.mainloop()
