Here's an English README that you can upload to GitHub for your project:
Airplane Battle Game with Q-learning
This project is a simple yet engaging airplane battle game developed using Python and Pygame. It incorporates Q-learning, a type of reinforcement learning, to enable the player's airplane to learn and improve its strategy over time.
Features
Dynamic Game Environment: The game features a dynamic environment where enemies (airplanes) drop from the top and move downwards.
Q-learning Integration: The game uses Q-learning to train the player's airplane to make better decisions and improve its survival skills.
State Discretization: Continuous state spaces are discretized to simplify the learning process.
Dynamic Epsilon Adjustment: The exploration rate (epsilon) is dynamically adjusted to balance exploration and exploitation.
Training Visualization: The training process includes visualization of the learning progress through a score plot.
How to Play
Controls: Use the left and right arrow keys to move the player's airplane.
Objective: Avoid enemy airplanes and survive as long as possible to achieve a higher score.
Getting Started
Requirements
Python 3.x
Pygame library
NumPy library
Matplotlib library (for visualization)
You can install the required libraries using pip:
pip install pygame numpy matplotlib

Navigate to the project directory:
cd airplane-battle

Run the game:
python game.py
Code Overview
Game Initialization
The game initializes the Pygame environment, sets up the window size, and loads the necessary image resources.

import pygame
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Set game window size
WIDTH, HEIGHT = 480, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Airplane Battle - Q-learning")

# Load image resources
current_dir = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_IMG = pygame.image.load(os.path.join(current_dir, "assets/background.png"))
PLAYER_IMG = pygame.image.load(os.path.join(current_dir, "assets/player.png"))
ENEMY_IMG = pygame.image.load(os.path.join(current_dir, "assets/enemy.png"))

# Adjust image sizes
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (50, 50))
ENEMY_IMG = pygame.transform.scale(ENEMY_IMG, (50, 50))
Player and Enemy Classes
The Player and Enemy classes handle the movement and updates of the player's airplane and enemy airplanes, respectively.
Python复制
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
Game Environment and Q-learning Agent
The PlaneWarEnv class represents the game environment, and the QLearningAgent class implements the Q-learning algorithm.

class PlaneWarEnv:
    def __init__(self):
        self.player = Player()
        self.enemies = []
        self.score = 0
        self.last_score_update = time.time()
        self.action_space = [0, 1, 2]  # 0: No move, 1: Left, 2: Right

    def reset(self):
        self.player.rect.center = (WIDTH // 2, HEIGHT - 50)
        self.enemies.clear()
        self.score = 0
        self.last_score_update = time.time()
        return self.get_state()

    def get_state(self):
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

        if action == 1:
            self.player.move("LEFT")
        elif action == 2:
            self.player.move("RIGHT")

        if random.randint(1, 50) == 1:
            self.enemies.append(Enemy())

        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.rect.top > HEIGHT:
                self.enemies.remove(enemy)
            if enemy.rect.colliderect(self.player.rect):
                reward = -1
                done = True

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
Training and Playing the Game
The train_agent function trains the Q-learning agent, and the play_game function allows you to play the game.

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
            pygame.time.delay(50)
        scores.append(env.score)
        print(f"Episode {episode + 1}, Score: {env.score}, Epsilon: {agent.epsilon}")
        agent.epsilon = max(0.01, agent.epsilon * 0.99)

    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.show()

# Create environment and agent
env = PlaneWarEnv()
agent = QLearningAgent(env)

# Train agent
train_agent(env, agent)

# Start game
def play_game(env, agent):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.delay(50)

# Run game
play_game(env, agent)

pygame.quit()
Conclusion
This project demonstrates how Q-learning can be applied to a simple game to improve decision-making over time. By discretizing the state space and dynamically adjusting the exploration rate, the player's airplane learns to survive longer and achieve higher scores.
Feel free to explore, modify, and enhance the project further.
