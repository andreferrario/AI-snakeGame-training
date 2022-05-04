import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomicita
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
        self.trainer = None


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #pericolo straight
            (dir_l and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_u)),


            # pericolo destra
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # pericolo sinistra
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)),

            # mosse
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # posizione cibo
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def ricorda(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, action, reward, next_states, done = zip(*mini_sample)
        self.trainer.train_step(states, action, reward, next_states, done)

    def train_short_memory(self,state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games

        final_move = [0, 0, 0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []

    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # risultati precedenti
        state_old = agent.get_state(game)


        # azione
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        new_state = agent.get_state(game)

        # allena memoria corta
        agent.train_short_memory(state_old, final_move, reward, new_state, done)

        agent.ricorda(state_old, final_move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save

            print('Partite: ', agent.n_games, 'Punteggio: ', score, 'Record: ', record)
            
if __name__ == '__main__':
    train()

