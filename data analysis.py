import pandas as pd
import matplotlib.pyplot as plt

# 1. 로그 파일 로드
try:
    df = pd.read_csv("agent_log_training.csv")
except FileNotFoundError:
    print("오류: 요약 로그 파일을 찾을 수 없습니다.")
    exit()

# 2. 후반부 안정화된 성능 확인
# "학습이 완료된" 마지막 1000개 에피소드의 평균 성능
stable_performance = df.iloc[-1000:]
print("--- 1단계 (Q-테이블) 최종 성능 ---")
print(f"평균 총 보상: {stable_performance['Total_Reward'].mean():.2f}")
print(f"평균 아이템 획득: {stable_performance['Items_Collected'].mean():.2f}")
print(f"평균 틱 수: {stable_performance['Total_Ticks'].mean():.2f}")
print(f"탈출 성공률: {stable_performance[stable_performance['End_Reason'] == 'Escape'].shape[0] / 1000.0 * 100.0:.1f}%")

# 3. (시각화 1) 학습 곡선 그리기
# 100개 에피소드 단위로 평균을 냄 (Moving Average)
df['Reward_MA'] = df['Total_Reward'].rolling(window=100).mean()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Reward_MA'], label='Total Reward (100-episode MA)')
plt.title('Q-Table Learning Curve (Baseline)')
plt.xlabel('Episode')
plt.ylabel('Average Total Reward')
plt.legend()
plt.grid(True)
plt.savefig('q_table_learning_curve.png') # 이미지 파일로 저장
print("학습 곡선(q_table_learning_curve.png) 저장 완료.")

# 4. (시각화 2) 아이템 획득 수
df['Items_MA'] = df['Items_Collected'].rolling(window=100).mean()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Items_MA'], label='Items Collected (100-episode MA)', color='orange')
plt.title('Q-Table Item Collection Over Time')
plt.xlabel('Episode')
plt.ylabel('Average Items Collected')
plt.legend()
plt.grid(True)
plt.savefig('q_table_item_collection.png') # 이미지 파일로 저장
print("아이템 획득 곡선(q_table_item_collection.png) 저장 완료.")