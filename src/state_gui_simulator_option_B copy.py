import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os
import time
import threading

# --------------------------------------------------
# [1] 경로 설정
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
project_root = os.path.dirname(current_dir)               # 프로젝트 루트
data_path = os.path.join(project_root, "data", "sensor_predictions_only.csv")
model_path = os.path.join(project_root, "results", "logistic_state_classifier.pkl")

# --------------------------------------------------
# [2] 데이터 및 모델 로딩
# --------------------------------------------------
df = pd.read_csv(data_path)
model = joblib.load(model_path)

# --------------------------------------------------
# [3] GUI 초기 설정
# --------------------------------------------------
root = tk.Tk()
root.title("📊 실시간 센서 상태 분류 시뮬레이터")
root.geometry("1000x500")

style = ttk.Style()
style.theme_use("default")

# 상태별 색상 태그 설정
style.configure("Treeview", rowheight=25)
tree = ttk.Treeview(root, show="headings")
tree.pack(expand=True, fill=tk.BOTH)

# 열 정의
columns = list(df.columns) + ["Predicted State"]
tree["columns"] = columns
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor="center")

# Scrollbar 추가
scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# 상태값에 따라 색상 강조
tree.tag_configure("state_0", background="#f0f0f0")  # 회색
tree.tag_configure("state_1", background="#d0f5d0")  # 연녹
tree.tag_configure("state_2", background="#d0e0ff")  # 연파랑
tree.tag_configure("state_3", background="#f8d0d0")  # 연빨강

# --------------------------------------------------
# [4] 시뮬레이션 함수
# --------------------------------------------------
def simulate_predictions():
    for idx, row in df.iterrows():
        features = row.values.reshape(1, -1)
        predicted_state = model.predict(features)[0]

        values = list(row.values) + [predicted_state]
        tag = f"state_{predicted_state}"

        item = tree.insert("", tk.END, values=values, tags=(tag,))
        tree.see(item)  # 자동 스크롤

        time.sleep(1)

# --------------------------------------------------
# [5] 스레드로 실행
# --------------------------------------------------
def start_simulation():
    start_btn.config(state="disabled")
    thread = threading.Thread(target=simulate_predictions)
    thread.daemon = True
    thread.start()

# --------------------------------------------------
# [6] 시작 버튼
# --------------------------------------------------
start_btn = ttk.Button(root, text="▶ 시뮬레이션 시작", command=start_simulation)
start_btn.pack(pady=10)

# --------------------------------------------------
# [7] GUI 실행
# --------------------------------------------------
root.mainloop()
