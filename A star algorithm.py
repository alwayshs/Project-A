import pygame

pygame.init

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) #화면 크기 설정
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)

CELL_SIZE = 40

# 0: 길, 1: 벽
grid = [
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
]

# 시작점 (row, col)
start = (0, 0)

# 도착점 (row, col)
end = (9, 14)

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent      # 이 노드를 발견한 부모 노드
        self.position = position  # 현재 노드의 위치 (row, col)

        self.g = 0  # 시작점부터 현재 노드까지의 비용
        self.h = 0  # 현재 노드부터 도착점까지의 추정 비용 (휴리스틱)
        self.f = 0  # g와 h를 더한 총 비용 (f = g + h)

    # 두 노드가 같은지 비교하기 위한 함수
    def __eq__(self, other):
        return self.position == other.position

open_list = []  # 열린 리스트
close_list = [] # 닫힌 리스트
path = None
path_found = False # 경로를 찾았는지 알려주는 플래그

# 시작 노드와 도착 노드를 생성
start_node = Node(None, start)
end_node = Node(None, end)

# open_list에 시작 노드를 추가
open_list.append(start_node)

# 휴리스틱 코스트 함수
def heuristic(current_node, end_node):
    # 현재 노드와 도착점 노드의 위치
    (x1, y1) = current_node.position
    (x2, y2) = end_node.position

    # 대각선 거리를 이용한 휴리스틱 (Diagonal Distance)
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    weight = 1.1 # << 가중치 추가 (1.1 ~ 1.5)
    cost = (10 * max(dx, dy) + 4 * min(dx, dy)) * weight
    
    return cost

def astar_step():
    global path, path_found # 전역 변수 선언

    if not open_list or path_found:
        return
    
    # 1. 가장 좋은 노드 선택하기
    # open_list에 있는 노드 중 f 비용이 가장 낮은 노드를 찾는다.
    current_node = open_list[0]
    current_index = 0
    for index, item in enumerate(open_list):
        if item.f < current_node.f:
            current_node = item
            current_index = index

    # 2. 선택된 노드를 open_list에서 빼고, close_list에 추가
    open_list.pop(current_index)
    close_list.append(current_node)

    # 3. 목표에 도달했는지 확인
    # 현재 노드가 도착점이면, 경로를 역추적해서 반환하고 종료
    if current_node == end_node:
        temp_path = []
        current = current_node
        while current is not None:
            temp_path.append(current.position)
            current = current.parent
            
        path = temp_path[::-1]
        path_found = True
        return # astar_step 함수 종료

    # 4. 이웃 노드 생성 및 탐색
    children = []
    # 8방향에 대해 (상, 하, 좌, 우, 대각선)
    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        
        # 노드 위치 계산
        node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

        # 맵 범위 안에 있는지 확인
        if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[0]) - 1) or node_position[1] < 0:
            continue

        # 벽(장애물)이 아닌지 확인
        if grid[node_position[0]][node_position[1]] != 0:
            continue

        # 새로운 자식 노드 생성
        new_node = Node(current_node, node_position)
        children.append(new_node)

    # 5. 자식 노드들을 순회하며 비용 계산 및 리스트에 추가
    for child in children:

        # 자식 노드가 close_list에 있다면 건너뛰기
        if child in close_list:
            continue

        # movement_cost를 10 또는 14로 설정하는 로직
        movement_cost = 10  # 기본값은 직선 비용으로 설정
        
        # 현재 노드와 자식 노드의 x, y 좌표가 모두 다른지 확인
        if current_node.position[0] != child.position[0] and current_node.position[1] != child.position[1]:
            movement_cost = 14 # x, y 좌표가 모두 다르면 대각선 이동

        # g, h, f 비용 계산
        child.g = current_node.g + movement_cost
        child.h = heuristic(child, end_node)
        child.f = child.g + child.h

        # 자식 노드가 이미 open_list에 있고, g 비용이 더 높은지 확인하는 로직
        is_better_path = True
        for open_node in open_list:
            if child == open_node and child.g >= open_node.g:
                is_better_path = False
                break
        
        if not is_better_path:
            continue

        # 위 모든 조건에 해당하지 않으면, open_list에 자식 노드 추가
        open_list.append(child)

# 메인 게임 루프를 위한 변수
running = True 

while running: #게임 루프
    screen.fill(BLACK) #단색으로 채워 화면 지우기

    #변수 업데이트

    # 모든 이벤트 처리
    for event in pygame.event.get():  
        if event.type == pygame.QUIT:
            running = False # 이 변수가 False가 되면 다음 프레임부터 루프가 멈춤

    # A* 알고리즘 한 단계 실행
    if not path_found: # 경로를 아직 못 찾았을 때만 알고리즘 실행
        astar_step()

    #화면 그리기

    for row_index in range(len(grid)):
        for column_index in range(len(grid[0])):
            rect = (CELL_SIZE * column_index, CELL_SIZE * row_index, CELL_SIZE, CELL_SIZE)
            value = grid[row_index][column_index]
            if value == 1:  # 벽
                pygame.draw.rect(screen, GRAY, rect)
            else:  # 길
                pygame.draw.rect(screen, WHITE, rect, 1)

    # 열린 리스트 (파랑)
    for node in open_list:
        r, c = node.position
        rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0, 0, 255), rect)

    # 닫힌 리스트 (빨강)
    for node in close_list:
        r, c = node.position
        rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (255, 0, 0), rect)

    # 최종 경로 (초록)
    if path:
        for (r, c) in path:
            rect = (CELL_SIZE * c, CELL_SIZE * r, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (0, 255, 0), rect)

    pygame.display.update()
    clock.tick(3)  # 속도 조절 (10 FPS)

pygame.quit()
