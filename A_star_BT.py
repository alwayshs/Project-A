import heapq

class PathNode:
    def __init__(self, parent=None, position=None):
        self.parent = parent      # 이 노드를 발견한 부모 노드
        self.position = position  # 현재 노드의 위치 (row, col)

        self.g = 0  # 시작점부터 현재 노드까지의 비용
        self.h = 0  # 현재 노드부터 도착점까지의 추정 비용 (휴리스틱)
        self.f = 0  # g와 h를 더한 총 비용 (f = g + h)

    # 두 노드가 같은지 비교하기 위한 함수
    def __eq__(self, other):
        return self.position == other.position

# 휴리스틱 코스트 함수
def heuristic(current_node, end_node):
    # 현재 노드와 도착점 노드의 위치
    (x1, y1) = current_node.position
    (x2, y2) = end_node.position

    # 대각선 거리를 이용한 휴리스틱 (Diagonal Distance)
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    weight = 1.0 # 가중치
    cost = (10 * max(dx, dy) + 4 * min(dx, dy)) * weight
    
    return cost

def astar(start, end, grid):
    open_list = []  # 열린 리스트
    close_list = [] # 닫힌 리스트
    count = 0

    start_node = PathNode(None, start)
    end_node = PathNode(None, end)

    heapq.heappush(open_list, (start_node.f, count, start_node))
    count += 1
    
    while len(open_list) > 0:

        current_f, _, current_node = heapq.heappop(open_list)

        if current_node in close_list:
            continue

        close_list.append(current_node)

        # 현재 노드가 도착점이면, 경로를 역추적해서 반환하고 종료
        if current_node == end_node:
            temp_path = []
            current = current_node
            while current is not None:
                temp_path.append(current.position)
                current = current.parent
            
            return temp_path[::-1]

        children = []
        # 8방향에 대해 (상, 하, 좌, 우, 대각선)
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        
            # 노드 위치 계산
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # 맵 범위 안에 있는지 확인
            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[0]) - 1) or node_position[1] < 0:
                continue

            # 벽(장애물)이 아닌지 확인
            tile_value = grid[node_position[0]][node_position[1]]

            # 1. 벽(1)은 항상 피함
            if tile_value == 1:
                continue

            # 새로운 자식 노드 생성
            new_node = PathNode(current_node, node_position)
            children.append(new_node)

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
            for _, _, open_node in open_list:
                if child == open_node and child.g >= open_node.g:
                    is_better_path = False
                    break
        
            if not is_better_path:
                continue

            # 위 모든 조건에 해당하지 않으면, open_list에 자식 노드 추가
            heapq.heappush(open_list, (child.f, count, child))
            count += 1

    # while 루프가 끝날 때까지 경로를 못 찾으면 None 반환
    return None