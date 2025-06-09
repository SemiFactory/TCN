import tkinter as tk
from tkinter import ttk
import pandas as pd # 그래프 최근 50개만 나오도록
import joblib
import os
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

# sensor_cols: 7개 컬럼만 뽑기 (CSV에 “Predicted State”가 이미 없다면 df.columns 그대로 7개)
sensor_cols = list(df.columns)

# --------------------------------------------------
# [3] GUI 초기화
# --------------------------------------------------
root = tk.Tk()
root.title("🔬 실시간 OHT 실험실 대시보드")
root.geometry("2000x1000")     # 가로를 넉넉히 잡아야 7개의 그래프가 충분히 보입니다
style = ttk.Style()
style.theme_use("default")
style.configure("Treeview", rowheight=25)

# --------------------------------------------------
# [4] 전체 레이아웃 프레임
# --------------------------------------------------
#   ┌────────────────────────────────────────────────────────────────┐
#   │                       상단 영역 (7개 분할 그래프)              │
#   ├────────────────────────────────────────────────────────────────┤
#   │                     하단 영역 (테이블 + 상태 꺾은선)          │
#   └────────────────────────────────────────────────────────────────┘

# 상단 영역
top_frame = ttk.Frame(root)
top_frame.pack(side="top", fill=tk.BOTH, expand=False)

# 하단 영역
bottom_frame = ttk.Frame(root)
bottom_frame.pack(side="bottom", fill=tk.BOTH, expand=True)

# --------------------------------------------------
# [5] 상단 영역 구성: 1행×7열 서브플롯
# --------------------------------------------------
graph_frame = ttk.LabelFrame(top_frame, text="📉 센서별 실시간 변화 (7분할)")
graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# 1×7 subplot 생성
fig_sensor, axs_sensor = plt.subplots(nrows=1, ncols=7, figsize=(20, 3), sharey=False)
fig_sensor.tight_layout(pad=3.0)

# 초기 타이틀 설정
for i, col in enumerate(sensor_cols):
    axs_sensor[i].set_title(col)
    axs_sensor[i].set_xlabel("샘플")
    axs_sensor[i].set_ylabel("값")

# FigureCanvasTkAgg로 embedding
canvas_sensor = FigureCanvasTkAgg(fig_sensor, master=graph_frame)
canvas_sensor.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# --------------------------------------------------
# [6] 이동 창 크기 설정 (최근 N개 샘플만 표시)
# --------------------------------------------------
window_size = 50

# 센서 데이터 저장용 딕셔너리 & 인덱스
sensor_data = {col: [] for col in sensor_cols}
sensor_index = []

def update_sensor_graph():
    """
    매 스텝마다 호출되어 7개의 축(axs_sensor[i])을 갱신.
    단, 'window_size' 만큼의 최근 데이터만 슬라이싱하여 보여준다.
    """
    for ax in axs_sensor:
        ax.clear()

    # 슬라이싱 범위 계산
    idx_window = sensor_index[-window_size:]
    for i, col in enumerate(sensor_cols):
        data_window = sensor_data[col][-window_size:]
        axs_sensor[i].plot(idx_window, data_window, color='tab:blue')
        axs_sensor[i].set_title(col)
        axs_sensor[i].set_xlabel("샘플")
        axs_sensor[i].set_ylabel("값")
        axs_sensor[i].grid(True)
        if idx_window:
            axs_sensor[i].set_xlim(min(idx_window), max(idx_window))
        # y축 자동조절 (필요시 ylim 설정 가능)

    fig_sensor.tight_layout(pad=3.0)
    canvas_sensor.draw()

# --------------------------------------------------
# [7] 하단 영역 구성
#     - 좌: 센서 상태 테이블
#     - 우: 상태(State) 꺾은선 그래프
# --------------------------------------------------
# 7-1) 좌측 – 센서 테이블
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

# 예측 state에 따른 색상 태그
tree.tag_configure("state_0", background="#c8facc")  # 정상
tree.tag_configure("state_1", background="#fff5b1")  # 주의
tree.tag_configure("state_2", background="#ffcb9a")  # 경고
tree.tag_configure("state_3", background="#ff8a8a")  # 위험

# 7-2) 우측 – 상태 꺾은선 그래프
state_frame = ttk.LabelFrame(bottom_frame, text="📊 상태(State) 추이")
state_frame.pack(side="right", fill=tk.BOTH, expand=True, padx=5, pady=5)

fig_state, ax_state = plt.subplots(figsize=(5, 4))
canvas_state = FigureCanvasTkAgg(fig_state, master=state_frame)
canvas_state.get_tk_widget().pack(fill=tk.BOTH, expand=True)

state_history = []

def update_state_graph():
    """
    매 스텝마다 state_history의 마지막 window_size 개만 그려서
    흐르는 형태의 그래프를 표시한다.
    """
    ax_state.clear()
    hist_window = state_history[-window_size:]
    if hist_window:
        x_vals = list(range(len(state_history) - len(hist_window), len(state_history)))
        ax_state.plot(x_vals, hist_window, marker='o', linestyle='-', color='red')
        ax_state.set_xlim(min(x_vals), max(x_vals))
    ax_state.set_title("State 값 변화")
    ax_state.set_xlabel("샘플")
    ax_state.set_ylabel("State")
    ax_state.set_yticks([0, 1, 2, 3])
    ax_state.grid(True)
    fig_state.tight_layout(pad=3.0)
    canvas_state.draw()

# --------------------------------------------------
# [8] 제어 버튼
# --------------------------------------------------
control_frame = ttk.Frame(root)
control_frame.pack(side="bottom", pady=10)

start_btn = ttk.Button(control_frame, text="▶ 실시간 탐색 시작")
stop_btn = ttk.Button(control_frame, text="⏹ 탐색 중단", state="disabled")
start_btn.pack(side="left", padx=10)
stop_btn.pack(side="left", padx=10)

# --------------------------------------------------
# [9] 시뮬레이션 로직
# --------------------------------------------------
stop_simulation = False

def simulate():
    global stop_simulation
    for idx, row in df.iterrows():
        if stop_simulation:
            break

        # ─────────────────────────────────────────────────────────────
        # (1) 로지스틱 모델 예측을 위한 DataFrame으로 변환
        features = pd.DataFrame([row.values], columns=sensor_cols)
        pred = model.predict(features)[0]   # 0,1,2,3 중 하나

        # ─────────────────────────────────────────────────────────────
        # (2) 센서 그래프용 데이터 추가
        sensor_index.append(idx)
        for col in sensor_cols:
            sensor_data[col].append(row[col])
        update_sensor_graph()

        # ─────────────────────────────────────────────────────────────
        # (3) State 히스토리에 추가 & 상태 꺾은선 그래프 갱신
        state_history.append(pred)
        update_state_graph()

        # ─────────────────────────────────────────────────────────────
        # (4) 테이블에 행 추가
        try:
            values = list(row.values) + [pred]
            tag = f"state_{pred}"
            tree.insert("", tk.END, values=values, tags=(tag,))
            tree.see(tree.get_children()[-1])
        except tk.TclError:
            # GUI가 닫히는 등의 예외 발생 시 루프 탈출
            break

        time.sleep(1)

    # 시뮬레이션 종료 후 버튼 상태 복원
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
# [10] GUI 메인루프 실행
# --------------------------------------------------
root.mainloop()
