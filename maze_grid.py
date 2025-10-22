import random
import csv

WIDTH = 50
HEIGHT = 50
PATH = 0
WALL = 1
ITEM = 2

# 아이템 생성 확률 (경로 칸 중 5%를 아이템으로 변경)
ITEM_PROBABILITY = 0.005

# 모든 칸을 벽(1)으로 채운 50x50 그리드 생성
grid = [[WALL for _ in range(WIDTH)] for _ in range(HEIGHT)]

stack = []
visited = set()

# 시작점
start_x = random.randrange(0, WIDTH, 2)
start_y = random.randrange(0, HEIGHT, 2)

grid[start_y][start_x] = PATH
stack.append((start_x, start_y))
visited.add((start_x, start_y))

while stack:
    cx, cy = stack[-1]  # 현재 위치 (스택의 top)
    
    # 방문 가능한 이웃 칸 탐색 (2칸씩 이동)
    neighbors = []
    # (dx, dy)는 (동, 서, 남, 북) 방향으로 2칸 이동
    for (dx, dy) in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
        nx, ny = cx + dx, cy + dy
        
        # 그리드 범위 내에 있고 아직 방문하지 않았다면
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and (nx, ny) not in visited:
            neighbors.append((nx, ny))
            
    if neighbors:
        # 방문할 이웃이 있다면
        nx, ny = random.choice(neighbors)  # 이웃 중 하나를 무작위로 선택
        
        # 현재 위치와 다음 위치 사이의 벽을 허뭄 (PATH, 0으로 설정)
        # (nx - cx) // 2 는 dx를 2로 나눈 값 (즉, 1 또는 -1)
        wx, wy = cx + (nx - cx) // 2, cy + (ny - cy) // 2
        grid[wy][wx] = PATH
        
        # 다음 위치를 길(PATH, 0)로 설정
        grid[ny][nx] = PATH
        
        # 다음 위치를 방문처리하고 스택에 추가
        visited.add((nx, ny))
        stack.append((nx, ny))
    else:
        # 방문할 이웃이 더 이상 없다면 (막다른 길)
        stack.pop()  # 스택에서 제거 (되돌아가기)

for y in range(HEIGHT):
    for x in range(WIDTH):
        # 현재 칸이 길(0)이라면
        if grid[y][x] == PATH:
            # 설정된 확률(ITEM_PROBABILITY)에 따라 아이템(2)으로 변경
            if random.random() < ITEM_PROBABILITY:
                grid[y][x] = ITEM

try:
    with open('maze_grid.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(grid)
    print("50x50 미로 그리드 생성 완료. 'maze_grid.csv' 파일로 저장되었습니다.")
    print("0: 길, 1: 벽, 2: 아이템")
except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")