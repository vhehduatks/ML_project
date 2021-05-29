import numpy as np
import pandas as pd
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 15)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
GRAY = (240,240,240)
RED = (200,0,0)
GREEN = (51, 204, 51)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w #스크린의 가로
        self.h = h #스크린의 높이
        # init display
        self.display = pygame.display.set_mode((self.w, self.h)) #스크린의 설정
        pygame.display.set_caption('Snake') #스크린의 타이틀
        self.clock = pygame.time.Clock() #fps설정을 위해 필요
        
        # # init game state
        # self.direction = Direction.RIGHT # 시작 방향의 결정
        
        # self.head = Point(self.w/2, self.h/2) # 뱀의 머리는 스크린의 중앙에 위치한 좌표값
        # self.snake = [self.head, #뱀의 구조는 뱀의 머리좌표,몸통1의 좌표, 몸통2의 좌표
        #               Point(self.head.x-BLOCK_SIZE, self.head.y),
        #               Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        # self.score = 0 #초기 점수는 0점
        # self.food = None #초기 food는 없음
        # self._place_food() #해당 함수로 food를 구현
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE # food의 x좌표는 0부터 (스크린의 x크기에서 블록의 크기를 뺀 다음 블록의 사이즈로 나눈 몫(을 구해야 좌표값이 블록단위로 떨어짐)다시 블록사이즈로 곱하므로써 정상 범위로 만들어줌 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE # 렌덤한 좌표로
        self.food = Point(x, y) #블록단위의 좌표를 재공
        if self.food in self.snake:#만약에 snake의 좌표값들중에 블록과 겹친다=뱀이 먹이를 먹는상황 + 뱀 위에 먹이가 생성된 상황)
            self._place_food()#새롭게 먹이를 재공
        
    def play_step(self,action):
        for event in pygame.event.get(): #이벤트처리(방향)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if event.type == pygame.KEYDOWN: #키입력 이벤트가 발생할경우
            #     if event.key == pygame.K_LEFT:#좌측키보드
            #         self.direction = Direction.LEFT#방향바꿈
            #     elif event.key == pygame.K_RIGHT:#이하동일
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN
        
        # 2. move
        self._move(action) # update the head = 헤드의 좌표를 키입력 이벤트에 따라서 바꾼다.
        self.snake.insert(0, self.head)# 바뀐 헤드(snake의 첫번째 item)를 snake에 업데이트한다. =헤드가 추가됨
        
        # 3. check if game over
        game_over = False
        if self.is_collision(): # 게임오버일 케이스
            game_over = True # 게임오버시키고
            return game_over, self.score #스코어 리턴
            
        # 4. place new food or just move
        if self.head == self.food: #먹이를 먹었을 경우
            self.score += 1 #스코어를 추가시키고
            self._place_food() #먹이를 배치함
        else:
            self.snake.pop() #먹이블록 이외의곳을 지날때는 맨 뒤에있는 꼬리를 pop 시킴=움직일때마다 헤드가 insert되므로 움직일때마다 꼬리가 pop 되어야 정상적인 뱀이 유지됨
        
        # 5. update ui and clock
        self._update_ui() #ui 를 업데이트 함= 지금까지의 정보를 바탕으로 스크린을 업데이트 함
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _drawGrid(self):
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, GRAY, rect, 1)

    def _update_ui(self):
        self.display.fill(WHITE)
        self._drawGrid()
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        state=self._get_state()
        score_text = font.render("score: " + str(self.score), True, BLACK)
        state_text = font.render("DS:{}  DR:{}  DL:{}  L:{}  R:{}  U:{}  D:{}  FL:{}  FR:{}  FU:{}  FD:{}".format(state[0],
                                                                                                                            state[1],
                                                                                                                            state[2],
                                                                                                                            state[3],
                                                                                                                            state[4],
                                                                                                                            state[5],
                                                                                                                            state[6],
                                                                                                                            state[7],
                                                                                                                            state[8],
                                                                                                                            state[9],
                                                                                                                            state[10]),True,BLACK)
        self.display.blit(score_text, [0, 0])
        self.display.blit(state_text, [0, 20])
        pygame.display.flip() # 화면의 업데이트
    
    def _get_state(self):
        L=self.direction==Direction.LEFT
        R=self.direction==Direction.RIGHT
        U=self.direction==Direction.UP
        D=self.direction==Direction.DOWN
        point_l = Point(self.head.x - 20, self.head.y)
        point_r = Point(self.head.x + 20, self.head.y)
        point_u = Point(self.head.x, self.head.y - 20)
        point_d = Point(self.head.x, self.head.y + 20)
        state=[
            # Danger straight
            (R and self.is_collision(point_r)) or 
            (L and self.is_collision(point_l)) or 
            (U and self.is_collision(point_u)) or 
            (D and self.is_collision(point_d)),

            # Danger right
            (U and self.is_collision(point_r)) or 
            (D and self.is_collision(point_l)) or 
            (L and self.is_collision(point_u)) or 
            (R and self.is_collision(point_d)),

            # Danger left
            (D and self.is_collision(point_r)) or 
            (U and self.is_collision(point_l)) or 
            (R and self.is_collision(point_u)) or 
            (L and self.is_collision(point_d)),
            
            # Move direction
            L,
            R,
            U,
            D,
            
            # Food location 
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y 
        ]
        return list(map(int,state))
        
    def _move(self, action):
        #action을 [직진,우회전,좌회전]->[1,0,0],[0,1,0],[0,0,1] 형식으로 받아들이므로

        #idx increase->clockwise ,idx decrease->counterclockwise
        dir_list=[Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx=dir_list.index(self.direction)

        #straight
        if np.array_equal(action,[1,0,0]):
            pass
        #right
        elif np.array_equal(action,[0,1,0]):
            idx=(idx+1)%4
            self.direction=dir_list[idx]
        #left
        else:
            idx=(idx-1)%4
            self.direction=dir_list[idx]

        if self.direction == Direction.RIGHT:
            self.head.x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            self.head.x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            self.head.y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            self.head.y -= BLOCK_SIZE

        # self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()