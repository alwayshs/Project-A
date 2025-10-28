import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 한글 폰트 설정을 위해 추가
import platform # OS 확인을 위해 추가

# --- 한글 폰트 설정 (Windows/macOS/Linux) ---

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
plt.rc('font', family=font_name)
plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지

# --- 파일 이름 설정 ---
Q_TABLE_LOG = "agent_log_training.csv" # 1단계(Q-테이블) 요약 로그
DQN_LOG = "agent_log_DQN_training.csv"       # 2단계(DQN) 요약 로그
ROLLING_WINDOW = 100 # 100개 에피소드 이동 평균

try:
    # 1. 두 개의 요약 로그 파일 로드
    df_q = pd.read_csv(Q_TABLE_LOG)
    df_dqn = pd.read_csv(DQN_LOG)
    print(f"'{Q_TABLE_LOG}' (Q-Table) 로그 {len(df_q)}줄 로드 완료.")
    print(f"'{DQN_LOG}' (DQN) 로그 {len(df_dqn)}줄 로드 완료.")

except FileNotFoundError as e:
    print(f"오류: 로그 파일 '{e.filename}'을 찾을 수 없습니다.")
    print("Q-테이블 및 DQN 학습 로그 파일이 모두 있는지 확인하세요.")
    exit()
except Exception as e:
    print(f"로그 로드 중 오류 발생: {e}")
    exit()

# 2. (시각화 1) 학습 곡선 비교 (Reward)
plt.figure(figsize=(14, 7))

# Q-Table의 이동 평균 계산
df_q['Reward_MA'] = df_q['Total_Reward'].rolling(window=ROLLING_WINDOW).mean()
plt.plot(df_q.index, df_q['Reward_MA'], label='Q-Table (베이스라인)')

# DQN의 이동 평균 계산
df_dqn['Reward_MA'] = df_dqn['Total_Reward'].rolling(window=ROLLING_WINDOW).mean()
plt.plot(df_dqn.index, df_dqn['Reward_MA'], label='DQN (개선 모델)', linestyle='--')

plt.title('DQN vs Q-Table: 학습 곡선 비교 (총 보상)', fontsize=16)
plt.xlabel('에피소드')
plt.ylabel(f'{ROLLING_WINDOW}-에피소드 이동 평균 총 보상')
plt.legend()
plt.grid(True)
plt.savefig('reward_comparison.png')
print("'reward_comparison.png' (보상 비교 그래프) 저장 완료.")


# 3. (시각화 2) 아이템 획득 수 비교
plt.figure(figsize=(14, 7))

# Q-Table의 이동 평균 계산
df_q['Items_MA'] = df_q['Items_Collected'].rolling(window=ROLLING_WINDOW).mean()
plt.plot(df_q.index, df_q['Items_MA'], label='Q-Table (베이스라인)')

# DQN의 이동 평균 계산
df_dqn['Items_MA'] = df_dqn['Items_Collected'].rolling(window=ROLLING_WINDOW).mean()
plt.plot(df_dqn.index, df_dqn['Items_MA'], label='DQN (개선 모델)', linestyle='--')

plt.title('DQN vs Q-Table: 아이템 획득 수 비교', fontsize=16)
plt.xlabel('에피소드')
plt.ylabel(f'{ROLLING_WINDOW}-에피소드 이동 평균 아이템 획득 수')
plt.legend()
plt.grid(True)
plt.savefig('item_comparison.png')
print("'item_comparison.png' (아이템 비교 그래프) 저장 완료.")


# 4. (분석) 최종 성능 수치 비교
# 각 모델의 마지막 1000개 에피소드 평균 성능
stable_q = df_q.iloc[-1000:]
stable_dqn = df_dqn.iloc[-1000:]

# Escape 성공률 계산 함수
def calculate_escape_rate(df):
    total_episodes = len(df)
    if total_episodes == 0:
        return 0.0
    # 'Escaped' 또는 'Mission_Complete' 등 성공 상태 확인
    success_reasons = ['Escape'] 
    successful_escapes = df[df['End_Reason'].isin(success_reasons)].shape[0]
    return (successful_escapes / total_episodes) * 100.0

print("\n최종 성능 비교 (마지막 1000 에피소드 평균)")
print("| 지표                | Q-Table (베이스라인) | DQN (개선 모델)  |")
print("|---------------------|----------------------|------------------|")
print(f"| 평균 총 보상        | {stable_q['Total_Reward'].mean():<20.2f} | {stable_dqn['Total_Reward'].mean():<16.2f} |")
print(f"| 평균 아이템 획득    | {stable_q['Items_Collected'].mean():<20.2f} | {stable_dqn['Items_Collected'].mean():<16.2f} |")
print(f"| 평균 틱 수          | {stable_q['Total_Ticks'].mean():<20.2f} | {stable_dqn['Total_Ticks'].mean():<16.2f} |")
print(f"| 탈출 성공률 (%)     | {calculate_escape_rate(stable_q):<20.1f} | {calculate_escape_rate(stable_dqn):<16.1f} |")
print("-----------------------------------------------------------")

print("\nfinal_comparison.py 실행 완료")