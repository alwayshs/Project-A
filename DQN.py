import pygame
import enum
import numpy as np
import math
import A_star_BT
import csv
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

try:    
    grid = np.loadtxt("maze_grid.csv", delimiter = ',', dtype = int)
    test_grid = grid.copy()
except FileNotFoundError:
    print("오류: 'grid.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    pygame.quit()
    exit()
except Exception as e:
    print(f"오류: grid.csv 파일을 읽는 중 문제가 발생했습니다: {e}")
    pygame.quit()
    exit()

class Status(enum.Enum):
    SUCCESS = 1
    RUNNING = 2
    FAILED = 3

# DQN 신경망
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 리플레이 버퍼
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # 오래된 기억부터 자동 삭제

    def push(self, *args):
        """경험(Transition)을 저장합니다."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """학습을 위해 batch_size만큼의 경험을 무작위로 샘플링합니다."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, start_position, bt_actions_list, n_observations, n_actions):
        # AI의 현재 위치 (row, col)
        self.position = start_position
        
        # AI의 단기 기억 장소
        self.memory = {}
        
        # 현재 따라가고 있는 경로
        self.path = []

        # 맵 전체를 미확인 상태로 복사
        self.map = np.full_like(grid, 3)

        self.items_collected = 0

        # 행동을 리스트로 저장
        self.bt_actions = bt_actions_list

        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.gamma = 0.9

        self.n_actions = n_actions
        self.n_observations = n_observations

        # policy망, target망
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 2. 경험 보관소 생성
        self.buffer = ReplayBuffer(10000) # 1만 개의 경험을 저장
        
        # 3. 최적화 도구 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
    
    def update_exploration_map(self, grid):
        agent_r, agent_c = self.position
        radius = 2 # 5x5 시야

        for r in range(agent_r - radius, agent_r + radius + 1):
            for c in range(agent_c - radius, agent_c + radius + 1):
                # (r, c)가 grid 맵 범위 안에 있는지 확인
                if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                    # 실제 grid 값을 읽어서 map에 기록
                    if grid[r][c] == 1: # 실제 벽이면
                        self.map[r][c] = 1 # 내 지도에 '벽'으로 기록
                    elif grid[r][c] == 0: # 길이면
                        self.map[r][c] = 0 # 내 지도에 '길'로 기록
                    elif grid[r][c] == 2: # 아이템이면
                        self.map[r][c] = 2 # 내 지도에 '아이템'으로 기록

class BehaviorNode():
    def __init__(self, name = "Node"):
        self.name = name
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        
    def state(self, agent, grid):
        raise NotImplementedError
    
class Selector(BehaviorNode):
    def __init__(self, name): # 자식도 이름 부여
        super().__init__(name) # 부모에게도 이름을 전달하며 호출
    
    def state(self, agent, grid):
        for child in self.children:
            status = child.state(agent, grid)
            if status != Status.FAILED:
                return status
        return Status.FAILED
    
class Sequence(BehaviorNode):
    def __init__(self, name): # 자식도 이름 부여
        super().__init__(name) # 부모에게도 이름을 전달하며 호출
    
    def state(self, agent, grid):
        for child in self.children:
            status = child.state(agent, grid)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS
    
class IsItemInMemory(BehaviorNode):
    def state(self, agent, grid):
        if 'target_item' in agent.memory:
            # print("기억된 아이템이 있습니다.")
            return Status.SUCCESS
        else:
            # print("기억된 아이템이 없습니다.")
            return Status.FAILED
    
class FindItemNearby(BehaviorNode):
    def state(self, agent, grid):
        item_pos = find_item_in_sight(agent, grid)
        
        if item_pos:
            agent.memory['target_item'] = item_pos
            # print(f"새 아이템 발견: {item_pos}")
            return Status.SUCCESS
        else:
            return Status.FAILED
        
class MoveToItem(BehaviorNode):
    def state(self, agent, grid):
        target_pos = agent.memory.get('target_item')

        if not target_pos: return Status.FAILED
        
        if agent.position == target_pos:
            print(f"아이템 획득! 위치: {target_pos}")
            # 아이템 획득 (grid에서 아이템 제거)
            if 0 <= target_pos[0] < len(grid) and 0 <= target_pos[1] < len(grid[0]):
                 if grid[target_pos[0]][target_pos[1]] == 2: # 해당 위치가 아이템이면
                      grid[target_pos[0]][target_pos[1]] = 0 # 길(0)으로 변경 (줍기)
                      agent.items_collected += 1
                      print("맵에서 아이템 제거 완료.")
                 else:
                      print("경고: 목표 위치에 아이템이 없습니다.")

            # 기억 지우기
            agent.memory.pop('target_item', None)
            agent.memory.pop('current_target', None) # 이동 목표도 함께 제거
            agent.path = [] # 현재 경로도 초기화

            return Status.SUCCESS # 성공 반환
        else:
            moved = move_one_step(agent, target_pos)
            if moved:
                return Status.RUNNING
            else:
                return Status.FAILED
        
class IsUnexploredArea(BehaviorNode):
    def state(self, agent, grid):
        # agent의 탐험 지도에 0 (미탐험)이 하나라도 존재하는지 확인
        if np.any(agent.map == 3):
            return Status.SUCCESS # "미개척지가 존재한다" -> 성공
        else:
            return Status.FAILED # "미개척지가 없다" -> 실패

class NearestUnexploredArea(BehaviorNode):
    def state(self, agent, grid):
        if 'exploration_target' in agent.memory:
            return Status.SUCCESS

        queue = deque([agent.position])
        visited = {agent.position}

        while queue:
            current_pos = queue.popleft()

            # 현재 위치(current_pos)의 8방향 이웃을 탐색
            for move_x in range(-1, 2): 
                for move_y in range(-1, 2):
                    if move_x == 0 and move_y == 0:
                        continue # 자기 자신 제외

                    next_pos = (current_pos[0] + move_x, current_pos[1] + move_y)

                    # 맵 범위 확인
                    if not (0 <= next_pos[0] < len(agent.map) and 0 <= next_pos[1] < len(agent.map[0])):
                        continue
                    
                    # 이미 방문한 곳인지 확인
                    if next_pos in visited:
                        continue

                    neighbor_tile = agent.map[next_pos[0]][next_pos[1]]

                    # '미탐험(3)' 구역인가?
                    if neighbor_tile == 3:
                        agent.memory['exploration_target'] = next_pos
                        return Status.SUCCESS

                    # '알려진 길(0)' 또는 '아이템(2)'인가?
                    if neighbor_tile in [0, 2]:
                        # '알려진 길'이라면, 계속 탐색하기 위해 큐에 추가
                        visited.add(next_pos)
                        queue.append(next_pos)

        # 큐가 비었는데도 '3'을 못 찾음 (모든 맵 탐험 완료)
        return Status.FAILED

class Exploration(BehaviorNode):
    def state(self, agent, grid):
        target_pos = agent.memory.get('exploration_target')
        if not target_pos: return Status.FAILED # 목표 없으면 실패 추가

        if agent.position == target_pos:
            print(f"탐험 목표 도달! 위치: {target_pos}")
            # 기억 삭제
            agent.memory.pop('exploration_target', None)
            agent.memory.pop('current_target', None) # 이동 목표도 함께 제거
            agent.path = [] # 현재 경로도 초기화
            return Status.SUCCESS # 성공 반환
        else:
            moved = move_one_step(agent, target_pos)
            if moved:
                return Status.RUNNING
            else:
                 # 경로 계산 실패 등의 이유로 이동 실패 시, 목표 재설정 유도
                 agent.memory.pop('exploration_target', None)
                 agent.memory.pop('current_target', None)
                 agent.path = []
                 return Status.FAILED

class Exit(BehaviorNode):
    def state(self, agent, grid):
        end_point = agent.memory.get('end_point')
        
        if agent.position != end_point:
            move_one_step(agent, end_point)
            return Status.RUNNING
        else:
            return Status.SUCCESS

def find_item_in_sight(agent, grid):
    agent_r, agent_c = agent.position # agent.position row, column 할당
    
    radius = 2 # 5*5 반경 설정

    nearest_item_pos = None
    min_distance = math.inf

    for r in range(agent_r - radius, agent_r + radius + 1):
        for c in range(agent_c - radius, agent_c + radius + 1):

            # (r, c)가 grid 맵 범위 안에 있는지 확인
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                # 현재 칸이 아이템(2)인지 확인
                if grid[r][c] == 2:
                    # 거리 계산
                    distance = abs(agent_r - r) + abs(agent_c - c)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_item_pos = (r, c)

    return nearest_item_pos

def get_q_state(agent, grid):
    """Q-러닝을 위한 현재 상태(S)를 PyTorch 텐서로 반환합니다."""
    s1 = 'target_item' in agent.memory
    s2 = find_item_in_sight(agent, grid) is not None
    s3 = np.any(agent.map == 3)
    s4 = not np.any(grid == 2)
    
    # 튜플 -> 리스트(float) -> 텐서
    state_list = [float(s1), float(s2), float(s3), float(s4)]
    return torch.tensor(state_list, dtype=torch.float32).unsqueeze(0)

def get_current_state_for_logging(agent, chosen_action_node, grid):
    """CSV 로깅을 위한 현재 상태 문자열을 반환합니다."""
    
    # Q-러닝이 선택한 행동의 이름을 가져옴
    action_name = chosen_action_node.name if chosen_action_node else "Idle"
    
    # Q-러닝의 목표 추적
    target = "None"
    if action_name == "기억된 아이템 획득":
        target = str(agent.memory.get('target_item'))
    elif action_name == "새로운 아이템 탐색":
        target = str(agent.memory.get('target_item'))
    elif action_name == "탐험 절차":
        target = str(agent.memory.get('exploration_target'))
    elif action_name == "탈출":
        target = str(agent.memory.get('end_point'))

    # S4 (탈출 조건) 확인
    is_escaping = not np.any(grid == 2)
    if is_escaping and action_name != "탈출":
        # 만약 아이템이 없는데 다른 행동을 하고 있다면 "탈출 시도 중"으로 간주
        action_name = "Escaping_Phase"

    return action_name, target

def move_one_step(agent, target_pos):
    # 경로가 비었거나, 기억된 목표와 현재 목표가 다르면 경로 재계산
    if not agent.path or agent.memory.get('current_target') != target_pos:
        print("agent.position:", agent.position)
        print("target:", agent.memory.get('exploration_target'))
        print("end_point:", agent.memory.get('end_point'))
        
        # A* 호출
        new_path = A_star_BT.astar(agent.position, target_pos, agent.map)

        if new_path:
            agent.path = new_path[1:] 
            # 현재 목표 지점을 기억
            agent.memory['current_target'] = target_pos 
        else:
            agent.path = [] # 경로 초기화
            agent.memory.pop('current_target', None) # 목표 제거
            return False # 길찾기 실패
        
    if agent.path:
        # 1. 다음 이동할 위치를 미리 확인
        next_pos = agent.path[0]
        
        # 2. 다음 위치가 현재 맵에서 벽(1)인지 확인
        if agent.map[next_pos[0]][next_pos[1]] == 1:
            # 3. 벽이라면, 현재 경로는 더 이상 유효하지 않음
            print(f"경로가 막힘! {next_pos}는 벽입니다. 경로를 재탐색합니다.")
            agent.path = [] # 경로 비우기
            agent.memory.pop('current_target', None) # 목표 비우기
            return False # 이동 실패 (-> 다음 틱에서 재계산 유도)
        # 경로 리스트에서 다음 위치를 하나 꺼냄
        next_pos = agent.path.pop(0)
        agent.position = next_pos
        return True # 이동 성공
    else:
        return False # 이동할 경로 없음
    
def optimize_model(agent, batch_size):
    if len(agent.buffer) < batch_size:
        return # 버퍼에 데이터가 충분히 쌓일 때까지 학습 안 함
    
    # 1. 버퍼에서 경험 샘플링
    transitions = agent.buffer.sample(batch_size)
    batch = Transition(*zip(*transitions)) # (s, a, r, s') 묶음

    # 2. 데이터 텐서로 변환
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

    # 3. Q(s, a) 계산
    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    # 4. max Q(s', a) 계산
    next_state_values = torch.zeros(batch_size)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
    next_state_values[non_final_mask] = agent.target_net(next_state_batch).max(1)[0].detach()
    
    # 5. 기대 Q 값 (Bellman) 계산: Q = R + gamma * max Q(s', a)
    expected_state_action_values = (next_state_values * agent.gamma) + reward_batch

    # 6. 손실(Loss) 계산 및 역전파
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

# 행동 트리 정의
memory_item_sequence = Sequence("기억된 아이템 획득")
memory_item_sequence.add_child(IsItemInMemory("아이템 기억 확인"))
memory_item_sequence.add_child(MoveToItem("아이템으로 이동"))

find_item_sequence = Sequence("새로운 아이템 탐색")
find_item_sequence.add_child(FindItemNearby("아이템 찾기"))
find_item_sequence.add_child(MoveToItem("아이템으로 이동"))

explore_sequence = Sequence("탐험 절차")
explore_sequence.add_child(IsUnexploredArea("미개척지 존재 확인"))
explore_sequence.add_child(NearestUnexploredArea("가장 가까운 미개척지 설정"))
explore_sequence.add_child(Exploration("미개척지로 이동"))

escape_action = Exit("탈출")

bt_action_list = [
    memory_item_sequence, # A0
    find_item_sequence,   # A1
    explore_sequence,     # A2
    escape_action         # A3
]

# Pygame 초기화 (시각화 설정)
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
CELL_SIZE = 10
BLACK = (0, 0, 0)

# Q-러닝 학습 루프 설정
NUM_EPISODES = 10000 # 총 학습 에피소드 횟수
MAX_TICKS_PER_EPISODE = 5000 # 에피소드 당 최대 틱 (무한루프 방지)

# 학습 시: IS_TRAINING = True (시각화 없이)
# 테스트 시: IS_TRAINING = False (학습된 Q-테이블로 시각화)
IS_TRAINING = False 
DQN_MODEL_FILE = 'dqn_policy_net.pth'

N_ACTIONS = 4 # 행동 4개
N_OBSERVATIONS = 4 # 상태 4개
BATCH_SIZE = 128 # 복습(학습) 단위
TARGET_UPDATE = 10 # 10 에피소드마다 업데이트

# 에이전트 생성 (루프 밖에서 한 번만 생성)
start_pos = (0, 0)
my_agent = Agent(start_pos, bt_action_list, N_OBSERVATIONS, N_ACTIONS)
my_agent.memory['end_point'] = (len(test_grid) - 1, len(test_grid[0]) - 1)

# 학습 모드가 아닐 경우(테스트/시각화), 학습된 Q-테이블 로드
if not IS_TRAINING:
    my_agent.epsilon = 0 # 테스트 시에는 탐험(무작위 행동) 안 함
    try:
        my_agent.policy_net.load_state_dict(torch.load(DQN_MODEL_FILE))
        my_agent.target_net.load_state_dict(my_agent.policy_net.state_dict()) # 타겟망도 복사
        print(f"'{DQN_MODEL_FILE}'에서 학습된 DQN 모델을 로드했습니다.")
    except FileNotFoundError:
        print(f"경고: '{DQN_MODEL_FILE}'을 찾을 수 없습니다.")
else:
    print(f"{NUM_EPISODES} 에피소드의 학습을 시작합니다")


# CSV 로거 설정
csv_headers = []
try:
    # IS_TRAINING 값에 따라 다른 로그 파일 사용
    log_filename = 'agent_log_DQN_training.csv' if IS_TRAINING else 'agent_log_DQN_test.csv'
    csv_file = open(log_filename, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    print(f"'{log_filename}' 파일이 열렸습니다. 로깅을 시작합니다.")
except IOError as e:
    print(f"CSV 파일 열기 오류: {e}")
    csv_file = None

# 메인 학습/실행 루프
running = True
for episode in range(NUM_EPISODES):
    if not running:
        break

    # 1. 환경 리셋 (매 에피소드마다)
    test_grid = grid.copy() # 원본 맵에서 다시 복사
    my_agent.position = start_pos
    my_agent.memory = {'end_point': (len(test_grid) - 1, len(test_grid[0]) - 1)}
    my_agent.path = []
    my_agent.map = np.full_like(test_grid, 3)
    my_agent.items_collected = 0
    
    tick_count = 0
    episode_running = True
    episode_running = True
    total_reward = 0 # 에피소드 총 보상 누적 변수
    end_reason = "Unknown" # 에피소드 종료 사유

    # 2. 현재 상태(S) 정의
    state = get_q_state(my_agent, test_grid)
    chosen_action_node = None
    
    while episode_running:
        
        # Pygame 이벤트 처리 (IS_TRAINING = False 일 때만 의미 있음)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                episode_running = False
                running = False # 전체 루프 종료

        # 3. 에이전트 시야 업데이트
        my_agent.update_exploration_map(test_grid)

        # 5. 행동(Action) 선택 (DQN Epsilon-Greedy)
        action_index = 0
        if random.uniform(0, 1) < my_agent.epsilon and IS_TRAINING:
            action_index = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad():
                q_values = my_agent.policy_net(state) 
                action_index = q_values.max(1)[1].item() # Q값이 가장 높은 행동의 인덱스
            
        chosen_action_node = my_agent.bt_actions[action_index]

        # 6. 행동 실행 및 보상 계산
        # Q-러닝이 선택한 행동 노드를 실행
        status = chosen_action_node.state(my_agent, test_grid)
        
        reward = 0
        if status == Status.SUCCESS:
            if chosen_action_node == escape_action:
                reward = 500
                end_reason = "Escaped"
                episode_running = False # 에피소드 종료
            elif chosen_action_node in [memory_item_sequence, find_item_sequence]:
                reward = 100 # 아이템 획득
            elif chosen_action_node == explore_sequence:
                reward = 10 # 탐험
        elif status == Status.FAILED:
            reward = -50 # 잘못된 행동
        elif status == Status.RUNNING:
            reward = -1 # 시간 비용

        total_reward += reward
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
            
        # 7. 다음 상태 확인
        new_state = None
        if not episode_running or (status == Status.SUCCESS and chosen_action_node == escape_action) or tick_count + 1 > MAX_TICKS_PER_EPISODE:
            # 에피소드가 끝나는 조건(탈출, 타임아웃)이면 new_state = None
            new_state = None 
        else:
            new_state = get_q_state(my_agent, test_grid)

        # 8. 버퍼에 경험 저장
        action_tensor = torch.tensor([[action_index]], dtype=torch.long)
        my_agent.buffer.push(state, action_tensor, new_state, reward_tensor)

        # 9. 상태 업데이트
        state = new_state

        # 10. 훈련 (매 틱마다)
        if IS_TRAINING:
            optimize_model(my_agent, BATCH_SIZE)
        
        # CSV 로깅
        if csv_file and not IS_TRAINING:
            r, c = my_agent.position
            action, target = get_current_state_for_logging(my_agent, chosen_action_node, test_grid)
            status_name = status.name
            items_count = my_agent.items_collected
            explored_count = np.count_nonzero(my_agent.map != 3)
            total_tiles = my_agent.map.size
            explore_percent = (explored_count / total_tiles) * 100
            
            data_row = [episode, tick_count, r, c, action, target, status_name, reward, items_count, f"{explore_percent:.2f}%"]
            writer.writerow(data_row)
        
        tick_count += 1
        
        # 탈출 조건 (루프 종료)
        if status == Status.SUCCESS and chosen_action_node == escape_action:
            # print(f"에피소드 {episode}: 탈출 성공")
            episode_running = False
            end_reason = "Escape"
            
        if tick_count > MAX_TICKS_PER_EPISODE:
            # print(f"에피소드 {episode}: 타임아웃")
            episode_running = False
            end_reason = "Timeout"
            
        # 화면 그리기 (학습 중이 아닐 때만)
        if not IS_TRAINING:
            screen.fill(BLACK)

            # 탐험 지도 그리기
            for r in range(len(my_agent.map)):
                for c in range(len(my_agent.map[0])):
                    rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
                    value = my_agent.map[r][c]
                    if value == 1: # 벽
                        pygame.draw.rect(screen, (128, 128, 128), rect)
                    elif value == 0: # 탐험된 길
                         pygame.draw.rect(screen, (255, 255, 255), rect) 

            # 아이템 그리기 (실제 grid 기준)
            for r in range(len(test_grid)):
                 for c in range(len(test_grid[0])):
                     if test_grid[r][c] == 2: # 아이템이면
                         rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
                         pygame.draw.rect(screen, (0, 255, 0), rect) # 초록색

            # 에이전트 그리기
            agent_r, agent_c = my_agent.position
            agent_rect = (CELL_SIZE * agent_c, CELL_SIZE * agent_r, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (255, 0, 0), agent_rect) # 빨간색

            pygame.display.update()
            clock.tick(10)
            
    # 에피소드 종료 후 처리

    if csv_file and IS_TRAINING:
            explored_count = np.count_nonzero(my_agent.map != 3)
            total_tiles = my_agent.map.size
            explore_percent = (explored_count / total_tiles) * 100
            items_count = my_agent.items_collected
        
            summary_row = [episode, tick_count, total_reward, items_count, f"{explore_percent:.2f}%", end_reason]
            writer.writerow(summary_row)

            csv_file.flush()
    
    # 엡실론 값 업데이트 (학습 모드일 때만)
    if IS_TRAINING:
        if my_agent.epsilon > my_agent.epsilon_min:
            my_agent.epsilon *= my_agent.epsilon_decay
            
    if (episode + 1) % 100 == 0 and IS_TRAINING:
        print(f"에피소드 {episode + 1}/{NUM_EPISODES} 완료. (Epsilon: {my_agent.epsilon:.3f})")

        if csv_file and IS_TRAINING:
            explored_count = np.count_nonzero(my_agent.map != 3)
            total_tiles = my_agent.map.size
            explore_percent = (explored_count / total_tiles) * 100
            items_count = my_agent.items_collected
        
            summary_row = [episode, tick_count, total_reward, items_count, f"{explore_percent:.2f}%", end_reason]
            writer.writerow(summary_row)

        # 100 에피소드마다 DQN 모델 저장
        try:
            torch.save(my_agent.policy_net.state_dict(), DQN_MODEL_FILE)
            print(f"DQN 모델 중간 저장 완료 ({DQN_MODEL_FILE})")
        except Exception as e:
            print(f"DQN 모델 중간 저장 중 오류: {e}")

    # 타겟 업데이트
    if IS_TRAINING and episode % TARGET_UPDATE == 0:
        my_agent.target_net.load_state_dict(my_agent.policy_net.state_dict())
        print(f"에피소드 {episode}: 타겟 네트워크 업데이트 완료")
        
    # 테스트 모드(시각화)일 경우 1판만 하고 종료
    if not IS_TRAINING:
        print(f"테스트 에피소드 완료. 총 {my_agent.items_collected}개 아이템 획득.")
        running = False # 전체 루프 종료

# 루프 종료 후 파일 닫기
if csv_file:
    csv_file.close()
    print(f"로그 파일 '{log_filename}' 저장 완료.")

# 학습 모드였다면 DQN 모델 저장
if IS_TRAINING:
    try:
        torch.save(my_agent.policy_net.state_dict(), DQN_MODEL_FILE)
        print(f"학습 완료. DQN 모델을 '{DQN_MODEL_FILE}'에 저장했습니다.")
    except Exception as e:
        print(f"DQN 모델 저장 중 오류 발생: {e}")

pygame.quit()
print("시뮬레이션 종료.")