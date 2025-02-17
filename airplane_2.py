import pygame
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# 初始化 Pygame
pygame.init()

# 设置游戏窗口的宽和高
WIDTH, HEIGHT = 480, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("飞机大战 - Q-learning")

# 加载图像资源
current_dir = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_IMG = pygame.image.load(os.path.join(current_dir, "assets/background.png"))
PLAYER_IMG = pygame.image.load(os.path.join(current_dir, "assets/player.png"))
ENEMY_IMG = pygame.image.load(os.path.join(current_dir, "assets/enemy.png"))

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

    def reset(self):
        self.player.rect.center = (WIDTH // 2, HEIGHT - 50)
        self.enemies.clear()
        self.score = 0
        self.last_score_update = time.time()
        return self.get_state()

    def get_state(self):
        # 获取离散化后的状态
        player_x = self.player.rect.x
        enemy_positions = [enemy.rect.x for enemy in self.enemies]
        state = [player_x] + enemy_positions
        return self.discretize_state(state)

    def discretize_state(self, state, bins=10):
        discretized_state = []
        for value in state:
            discretized_value = int((value / WIDTH) * bins)
            discretized_state.append(min(discretized_value, bins - 1))
        return tuple(discretized_state)

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

        next_state = self.get_state()
        return next_state, reward, done, {}

    def render(self):
        screen.fill((0, 0, 0))
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

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
        self.q_table[(state, action)] += self.alpha * td_error

def train_agent(env, agent, episodes=100):
    scores = []
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
        scores.append(env.score)
        print(f"Episode {episode + 1}, Score: {env.score}, Epsilon: {agent.epsilon}")
        agent.epsilon = max(0.01, agent.epsilon * 0.99)  # 动态调整 epsilon

    # 绘制学习曲线
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.show()

# 创建环境和代理
env = PlaneWarEnv()
agent = QLearningAgent(env)

# 训练代理
train_agent(env, agent)

# 启动游戏
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

pygame.quit()