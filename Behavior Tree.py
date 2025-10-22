import pygame
import enum
import numpy as np
import math
import A_star_BT
from collections import deque
import csv

try:    
    grid = np.loadtxt("maze_grid.csv", delimiter = ',', dtype = int)
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

class Agent:
    def __init__(self, start_position):
        # AI의 현재 위치 (row, col)
        self.position = start_position
        
        # AI의 단기 기억 장소
        self.memory = {}
        
        # 현재 따라가고 있는 경로
        self.path = []

        # 맵 전체를 미확인 상태로 복사
        self.map = np.full_like(grid, 3)

        self.items_collected = 0
    
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
        
    def state(self):
        raise NotImplementedError
    
class Selector(BehaviorNode):
    def __init__(self, name): # 자식도 이름 부여
        super().__init__(name) # 부모에게도 이름을 전달하며 호출
    def state(self, agent):
        for child in self.children:
            status = child.state(agent)
            if status != Status.FAILED:
                return status
        return Status.FAILED
    
class Sequence(BehaviorNode):
    def __init__(self, name): # 자식도 이름 부여
        super().__init__(name) # 부모에게도 이름을 전달하며 호출
    def state(self, agent):
        for child in self.children:
            status = child.state(agent)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS
    
class IsItemInMemory(BehaviorNode):
    def state(self, agent):
        if 'target_item' in agent.memory:
            # print("기억된 아이템이 있습니다.")
            return Status.SUCCESS
        else:
            # print("기억된 아이템이 없습니다.")
            return Status.FAILED
    
class FindItemNearby(BehaviorNode):
    def state(self, agent):
        item_pos = find_item_in_sight(agent, grid)
        
        if item_pos:
            agent.memory['target_item'] = item_pos
            # print(f"새 아이템 발견: {item_pos}")
            return Status.SUCCESS
        else:
            return Status.FAILED
        
class MoveToItem(BehaviorNode):
    def state(self, agent):
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
    def state(self, agent):
        # agent의 탐험 지도에 0 (미탐험)이 하나라도 존재하는지 확인
        if np.any(agent.map == 3):
            return Status.SUCCESS # "미개척지가 존재한다" -> 성공
        else:
            return Status.FAILED # "미개척지가 없다" -> 실패

class NearestUnexploredArea(BehaviorNode):
    def state(self, agent):
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
    def state(self, agent):
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
    def state(self, agent):
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

def get_current_state_for_logging(agent):
    # 현재 에이전트의 행동과 목표를 문자열로 반환
    
    # 1순위: 탈출
    # np.any(grid == 2)는 "맵에 2(아이템)가 하나라도 있는가?"
    if not np.any(grid == 2):
        return "Escaping", str(agent.memory.get('end_point'))

    # 2순위: 아이템 획득
    if 'target_item' in agent.memory:
        return "MoveToItem", str(agent.memory.get('target_item'))

    # 3순위: 탐험
    if 'exploration_target' in agent.memory:
        return "Exploring", str(agent.memory.get('exploration_target'))
        
    # 4순위: 결정 중 또는 유휴 상태
    return "Idle/Deciding", "None"

def move_one_step(agent, target_pos):
    # 경로가 비었거나, 기억된 목표와 현재 목표가 다르면 경로 재계산
    if not agent.path or agent.memory.get('current_target') != target_pos:
        print("agent.position:", agent.position)
        print("target:", agent.memory.get('exploration_target'))
        print("end_point:", agent.memory.get('end_point'))
        
        # A* 호출
        new_path = A_star_BT.astar(agent.position, target_pos, agent.map)

        if new_path:
            # 첫 번째는 현재 위치이므로 제외하고 경로 저장
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

# 루트 노드 생성
root = Selector("최상위 의사결정")

# 1순위: '기억된' 아이템 획득 절차
memory_item_sequence = Sequence("기억된 아이템 획득")
memory_item_sequence.add_child(IsItemInMemory("아이템 기억 확인"))
memory_item_sequence.add_child(MoveToItem("아이템으로 이동"))

# 2순위: '새로운' 아이템 탐색 및 획득 절차
find_item_sequence = Sequence("새로운 아이템 탐색")
find_item_sequence.add_child(FindItemNearby("아이템 찾기"))
find_item_sequence.add_child(MoveToItem("아이템으로 이동"))

# 3순위: 탐험 절차
explore_sequence = Sequence("탐험 절차")
explore_sequence.add_child(IsUnexploredArea("미개척지 존재 확인"))
explore_sequence.add_child(NearestUnexploredArea("가장 가까운 미개척지 설정"))
explore_sequence.add_child(Exploration("미개척지로 이동"))

# 4순위: 탈출
escape_action = Exit("탈출")

# 루트에 자식들 추가
root.add_child(memory_item_sequence) # 1순위
root.add_child(find_item_sequence)   # 2순위
root.add_child(explore_sequence)     # 3순위
root.add_child(escape_action)        # 4순위

# --- Pygame 초기화 및 설정 ---
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
CELL_SIZE = 10

BLACK = (0, 0, 0)

csv_headers = ['Tick', 'Position_Row', 'Position_Col', 'Current_Action', 'Target', 'Node_Status', 'Items_Collected', 'Exploration_Percent']
try:
    csv_file = open('agent_log.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    print("'agent_log.csv' 파일이 열렸습니다. 로깅을 시작합니다.")
except IOError as e:
    print(f"CSV 파일 열기 오류: {e}")
    # 파일 열기에 실패하면 csv_file 변수가 생성되지 않음
    csv_file = None

# --- 에이전트 생성 ---
start_pos = (0, 0) # 시작 위치
my_agent = Agent(start_position = start_pos)
# 탈출 지점 메모리에 저장
my_agent.memory['end_point'] = (len(grid) - 1, len(grid[0]) - 1)

# --- 메인 게임 루프 ---
running = True
tick_count = 0

while running:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 1. 에이전트의 시야에 따라 탐험 지도 업데이트
    my_agent.update_exploration_map(grid)
    
    # 2. 행동 트리 실행
    root.state(my_agent)

    status = root.state(my_agent)

    if my_agent.position == my_agent.memory.get('end_point'):
        print("탈출 성공! 3초 후 프로그램을 종료합니다.")
        pygame.time.wait(3000) # 3초 대기 (성공 확인용)
        running = False        # 메인 루프 종료 플래그 설정

    if csv_file: 
        # 1. 현재 상태 가져오기
        r, c = my_agent.position
        action, target = get_current_state_for_logging(my_agent)
        
        # 추가 데이터 계산
        status_name = status.name
        items_count = my_agent.items_collected
        
        # (탐험된 타일 수 / 전체 타일 수) * 100
        explored_count = np.count_nonzero(my_agent.map != 3)
        total_tiles = my_agent.map.size
        explore_percent = (explored_count / total_tiles) * 100
        
        # 2. 데이터 한 줄로 만들기 (새 항목 추가)
        data_row = [tick_count, r, c, action, target, status_name, items_count, f"{explore_percent:.2f}%"]
        
        # 3. 파일에 쓰기
        writer.writerow(data_row)
    
    tick_count += 1

    # --- 화면 그리기 ---
    screen.fill(BLACK)

    # 탐험 지도 그리기
    for r in range(len(my_agent.map)):
        for c in range(len(my_agent.map[0])):
            rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
            value = my_agent.map[r][c]
            if value == 1: # 벽
                pygame.draw.rect(screen, (128, 128, 128), rect)
            elif value == 0: # 탐험된 길
                 pygame.draw.rect(screen, (255, 255, 255), rect) # 약간 어둡게
            # value == 3 (미탐험) 은 그냥 검은색 배경

    # 아이템 그리기
    for r in range(len(grid)):
         for c in range(len(grid[0])):
             if grid[r][c] == 2: # 아이템이면
                 rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
                 pygame.draw.rect(screen, (0, 255, 0), rect) # 초록색으로

    # 에이전트 그리기
    agent_r, agent_c = my_agent.position
    agent_rect = (CELL_SIZE * agent_c, CELL_SIZE * agent_r, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (255, 0, 0), agent_rect) # 빨간색으로

    pygame.display.update()
    clock.tick(100) # 속도 조절

# --- 루프 종료 후 파일 닫기 ---
if csv_file:
    csv_file.close()
    print("로그 파일 'agent_log.csv' 저장 완료.")

pygame.quit()
