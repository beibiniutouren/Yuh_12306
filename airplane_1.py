import pygame
import random
import time
import os

print("Current working directory:", os.getcwd())
# 初始化Pygame
pygame.init()

# 设置游戏窗口的宽和高
WIDTH, HEIGHT = 480, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("飞机大战 - Q-learning")

# 加载图像资源
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建图像文件的绝对路径
BACKGROUND_IMG = pygame.image.load(os.path.join(current_dir, "assets/background.png"))
PLAYER_IMG = pygame.image.load(os.path.join(current_dir, "assets/player.png"))
ENEMY_IMG = pygame.image.load(os.path.join(current_dir, "assets/enemy.png"))

# 调整图像大小
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (50, 50))
ENEMY_IMG = pygame.transform.scale(ENEMY_IMG, (50, 50))

class Player_start:
    def __init__(self):
        self.image = PLAYER_IMG
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        self.speed = 5

    def move(self, direction):
        if direction == "LEFT" and self.rect.left > 0:
            self.rect.x -= self.speed
        elif direction == "RIGHT" and self.rect.right < WIDTH:
            self.rect.x += self.speed

class Enemy_start:
    def __init__(self):
        self.image = ENEMY_IMG
        self.rect = self.image.get_rect(center=(random.randint(0, WIDTH), -50))
        self.speed = random.randint(2, 5)

    def update(self):
        self.rect.y += self.speed

def main():
    player = Player_start()
    enemies = []
    score = 0
    last_score_update = time.time()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move("LEFT")
        if keys[pygame.K_RIGHT]:
            player.move("RIGHT")

        # 更新敌机
        if random.randint(1, 50) == 1:
            enemies.append(Enemy_start())

        for enemy in enemies[:]:
            enemy.update()
            if enemy.rect.top > HEIGHT:
                enemies.remove(enemy)
            if enemy.rect.colliderect(player.rect):
                running = False  # 碰撞后游戏结束

        # 更新得分
        current_time = time.time()
        if current_time - last_score_update > 1:
            score += 1
            last_score_update = current_time

        # 绘制
        screen.blit(BACKGROUND_IMG, (0, 0))
        screen.blit(player.image, player.rect)
        for enemy in enemies:
            screen.blit(enemy.image, enemy.rect)

        # 显示得分
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

#Q-learning 部分
import pygame
import sys
import numpy as np
import random
import time

# 初始化 Pygame
pygame.init()

# 设置窗口大小
WIDTH, HEIGHT = 480, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("飞机大战")

# 加载图像资源
BACKGROUND_IMG = pygame.image.load("assets/background.png")
PLAYER_IMG = pygame.image.load("assets/player.png")
ENEMY_IMG = pygame.image.load("assets/enemy.png")

# 调整图像大小
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (50, 50))
ENEMY_IMG = pygame.transform.scale(ENEMY_IMG, (50, 50))

class Player:
    def __init__(self):
        self.image = PLAYER_IMG
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        self.speed = 5

    def move(self, direction):
        if direction == "LEFT" and self.rect.left > 0:
            self.rect.x -= self.speed
        elif direction == "RIGHT" and self.rect.right < WIDTH:
            self.rect.x += self.speed

class Enemy:
    def __init__(self):
        self.image = ENEMY_IMG
        self.rect = self.image.get_rect(center=(random.randint(0, WIDTH), -50))
        self.speed = random.randint(2, 5)

    def update(self):
        self.rect.y += self.speed

class PlaneWarEnv:
    def __init__(self):
        self.player = Player()
        self.enemies = []
        self.score = 0
        self.last_score_update = time.time()
        self.action_space = [0, 1, 2]  # 0: 不动, 1: 左移, 2: 右移
        self.observation_space = np.array([self.player.rect.x, self.player.rect.y])

    def reset(self):
        self.player.rect.center = (WIDTH // 2, HEIGHT - 50)
        self.enemies.clear()
        self.score = 0
        self.last_score_update = time.time()
        return self.observation_space

    def step(self, action):
        reward = 0
        done = False

        # 执行动作
        if action == 1:
            self.player.move("LEFT")
        elif action == 2:
            self.player.move("RIGHT")

        # 更新敌机
        if random.randint(1, 50) == 1:
            self.enemies.append(Enemy())

        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.rect.top > HEIGHT:
                self.enemies.remove(enemy)
            if enemy.rect.colliderect(self.player.rect):
                reward = -1
                done = True

        # 更新得分
        current_time = time.time()
        if current_time - self.last_score_update > 1:
            self.score += 1
            self.last_score_update = current_time
            reward = 1

        # 更新观察空间
        self.observation_space = np.array([self.player.rect.x, self.player.rect.y])

        return self.observation_space, reward, done, {}

    def render(self):
        screen.fill((0, 0, 0))  # 清除屏幕
        screen.blit(BACKGROUND_IMG, (0, 0))
        screen.blit(self.player.image, self.player.rect)
        for enemy in self.enemies:
            screen.blit(enemy.image, enemy.rect)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索概率
        self.q_table = {}

    def get_q_value(self, state, action):
        if (tuple(state), action) not in self.q_table:
            self.q_table[(tuple(state), action)] = 0.0
        return self.q_table[(tuple(state), action)]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            q_values = [self.get_q_value(state, action) for action in self.env.action_space]
            return self.env.action_space[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = self.choose_action(next_state)
        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
        td_error = td_target - self.get_q_value(state, action)
        self.q_table[(tuple(state), action)] += self.alpha * td_error

def train_agent(env, agent, episodes=5):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            env.render()
            pygame.time.delay(50)  # 控制训练速度
        print(f"Episode {episode + 1}, Score: {env.score}")

# 创建环境和代理
env = PlaneWarEnv()
agent = QLearningAgent(env)

# 训练代理
train_agent(env, agent)

pygame.quit()
sys.exit()

#启动游戏
def play_game(env, agent):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.delay(50)  # 控制游戏速度

# 运行游戏
play_game(env, agent)