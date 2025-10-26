import pandas as pd
import matplotlib.pyplot as plt

print("\nAI 행동 정밀 분석 (Test Log)")

try:
    df_test = pd.read_csv("agent_log_test.csv")
    print(f"테스트 로그(agent_log_test.csv) 로드 완료. (총 {len(df_test)} 틱)")
except FileNotFoundError:
    print("오류: agent_log_test.csv 파일을 찾을 수 없습니다.")
    exit()

# 1. (시각화) AI가 이동한 최종 경로
plt.figure(figsize=(10, 10))
plt.plot(df_test[' c'], df_test[' r'], label='Agent Path')
plt.title('Final Agent Path')
plt.xlabel('Column (X)')
plt.ylabel('Row (Y)')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis() # Y축 뒤집기
plt.savefig('final_agent_path.png')
print("AI 최종 경로(final_agent_path.png) 저장 완료.")


# 2. (분석) AI가 각 행동에 몇 틱을 소모했는가?
action_ticks = df_test[' action'].value_counts()
print("\nAI 행동(전략)별 소요 시간(틱)")
print(action_ticks)

# 3. (분석) AI의 전략적 결정 순서
# 'action'이 변경되는 순간만 추출
decision_changes = df_test[df_test[' action'].shift() != df_test[' action']]
print("\nAI의 전략적 결정 순서")
print(decision_changes[[' tick_count', ' action', ' target']])