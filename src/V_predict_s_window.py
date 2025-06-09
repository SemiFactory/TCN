import pandas as pd
import numpy as np
import os

# 연속된 시계열 구간을 찾는 함수
def find_continuous_sequences(df, interval_sec=1, tolerance=0.1, min_length=50):
    """
    df : timestamp, state 센서값들 컬럼을 가진 pandas DataFrame
    interval_sec : 기대하는 시간 간격 '초'
    tolerance : 허용 오차율 10%
    min_length : 시퀀스 최소 길이, 짧은 건 무시
    일정한 시간 간격을 갖는 연속된 시계열 구간을 찾습니다.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    lower = interval_sec * (1 - tolerance)
    upper = interval_sec * (1 + tolerance)

    sequences = []
    start_idx = 0
    for i in range(1, len(df)):
        if not (lower <= df.loc[i, 'time_diff'] <= upper):
            if i - start_idx >= min_length:
                sequences.append((start_idx, i - 1, df.loc[start_idx, 'timestamp'], df.loc[i - 1, 'timestamp'], i - start_idx))
            start_idx = i
    if len(df) - start_idx >= min_length:
        sequences.append((start_idx, len(df) - 1, df.loc[start_idx, 'timestamp'], df.loc[len(df) - 1, 'timestamp'], len(df) - start_idx))
    return sequences

def generate_future_state_windows(df, sequences, window_size=50, stride=1, future_offset= 18):
    """
    과거 시퀀스를 기반으로 미래 상태(state)를 예측하는 학습 데이터를 생성합니다.
    df : 원본 DataFrame (timestamp, state, 센서값)
    sequences : 위에서 찾은 시퀀스 리스트
    window_size : 과거 시퀀스 길이 (기본 50개 시점)
    stride : 윈도우 이동 간격 (기본 1)
    future_offset : 미래 예측 시점 설정 10초 뒤
    """
    X, y = [], []

    for start, end, _, _, _ in sequences:
        sub_df = df.iloc[start:end + 1].reset_index(drop=True)
        max_start = len(sub_df) - window_size - future_offset  # 미래 시점까지 확보
        for i in range(0, max_start + 1, stride):
            window = sub_df.iloc[i:i + window_size]
            future = sub_df.iloc[i + window_size + future_offset - 1]  # t+future_offset

            features = window.drop(columns=['timestamp', 'state']).values
            future_state = future['state']

            X.append(features)
            y.append(future_state)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y# x : 과거 센서 시퀀스 데이터 / y : 미래 state 값들

def save_windows(X, y):
    """
    생성된 데이터를 numpy 파일로 저장합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "VX_windows_predict.npy"), X)
    np.save(os.path.join(save_dir, "Vy_states_pridict.npy"), y)
    print(f"[✅] 저장 완료: {save_dir}")

if __name__ == "__main__":
    # CSV 파일 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "data", "V_f_t_d.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[❌] 파일이 존재하지 않습니다: {csv_path}")

    # 데이터 로드
    df = pd.read_csv(csv_path)

    # 연속된 시계열 구간 탐색
    sequences = find_continuous_sequences(df)

    # 미래 상태 예측용 윈도우 데이터 생성
    X, y = generate_future_state_windows(df, sequences, window_size=50, stride=1)

    # 파일 저장
    save_windows(X, y)
