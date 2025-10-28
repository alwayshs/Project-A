import pandas as pd
import matplotlib.pyplot as plt
import time

# --- 1. DQN 요약 로그 분석 ---

DQN_LOG_FILE = "agent_log_DQN_training.csv" # 분석할 DQN 요약 로그 파일
ROLLING_WINDOW = 100 # 이동 평균 윈도우 크기

try:
    # 1. DQN 요약 로그 파일 로드
    df_dqn = pd.read_csv(DQN_LOG_FILE)
    print(f"'{DQN_LOG_FILE}' (DQN) 로그 {len(df_dqn)}줄 로드 완료.")

except FileNotFoundError:
    print(f"오류: '{DQN_LOG_FILE}'을 찾을 수 없습니다.")
    print("DQN 학습을 먼저 실행하여 로그 파일을 생성했는지 확인하세요.")
    exit()
except Exception as e:
    print(f"DQN 로그 로드 중 오류 발생: {e}")
    exit()

# 2. (DQN) 후반부 안정화된 성능 확인
# 학습이 완료된 마지막 1000개 에피소드의 평균 성능
stable_performance = df_dqn.iloc[-1000:]
print("2단계 (DQN) 최종 성능")
print(f"평균 총 보상: {stable_performance['Total_Reward'].mean():.2f}")
print(f"평균 아이템 획득: {stable_performance['Items_Collected'].mean():.2f}")
print(f"평균 틱 수: {stable_performance['Total_Ticks'].mean():.2f}")
# 여기서는 'Escape'를 성공 기준으로 가정
escape_rate = 0.0
if 'Escape' in stable_performance['End_Reason'].unique():
    escape_rate = stable_performance[stable_performance['End_Reason'] == 'Escaped'].shape[0] / 1000.0 * 100.0
print(f"탈출 성공률: {escape_rate:.1f}%")

# 3. (시각화 1) DQN 학습 곡선 그리기
df_dqn['Reward_MA'] = df_dqn['Total_Reward'].rolling(window=ROLLING_WINDOW).mean()
plt.figure(figsize=(12, 6))
plt.plot(df_dqn.index, df_dqn['Reward_MA'], label=f'DQN Total Reward ({ROLLING_WINDOW}-episode MA)')
plt.title('DQN Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.legend()
plt.grid(True)
plt.savefig('dqn_learning_curve.png') # DQN 학습 곡선 이미지 저장
print("DQN 학습 곡선(dqn_learning_curve.png) 저장 완료.")

# 4. (시각화 2) DQN 아이템 획득 수
df_dqn['Items_MA'] = df_dqn['Items_Collected'].rolling(window=ROLLING_WINDOW).mean()
plt.figure(figsize=(12, 6))
plt.plot(df_dqn.index, df_dqn['Items_MA'], label=f'DQN Items Collected ({ROLLING_WINDOW}-episode MA)', color='orange')
plt.title('DQN Item Collection Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Items Collected')
plt.legend()
plt.grid(True)
plt.savefig('dqn_item_collection.png') # DQN 아이템 획득 곡선 이미지 저장
print("DQN 아이템 획득 곡선(dqn_item_collection.png) 저장 완료.")

print("\ndqn_analysis.py 실행 완료")