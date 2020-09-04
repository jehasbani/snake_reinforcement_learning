from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
import random
import numpy as np
import pandas as pd
import collections


class Agent():
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(activation='relu', input_dim=12, units=self.first_layer))
        model.add(Dense(activation='relu', units=self.second_layer))
        model.add(Dense(activation='relu', units=self.third_layer))
        model.add(Dense(activation='softmax', units=4))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def get_state(self, game, snake, food):

        snake_head = snake.get_head_position()

        danger_up = (snake.direction == snake.up and
                     ((((snake_head[0] + (snake.up[0] * game.gridsize)) % game.width),
                       (snake_head[1] + (snake.up[1] * game.gridsize)) % game.height) in snake.positions or
                      (snake_head[1] + (snake.up[1] * game.gridsize)) % game.height >= game.height - game.gridsize))
        danger_down = (snake.direction == snake.down and
                       ((((snake_head[0] + (snake.down[0] * game.gridsize)) % game.width),
                         (snake_head[1] + (snake.down[1] * game.gridsize)) % game.height) in snake.positions or
                        (snake_head[1] + (snake.down[1] * game.gridsize)) % game.height < game.gridsize))
        danger_left = (snake.direction == snake.left and
                       ((((snake_head[0] + (snake.left[0] * game.gridsize)) % game.width),
                         (snake_head[1] + (snake.left[1] * game.gridsize)) % game.height) in snake.positions or
                        (snake_head[0] + (snake.left[0] * game.gridsize)) % game.height < game.gridsize))
        danger_right = (snake.direction == snake.right and
                        ((((snake_head[0] + (snake.right[0] * game.gridsize)) % game.width),
                          (snake_head[1] + (snake.right[1] * game.gridsize)) % game.height) in snake.positions or
                         (snake_head[0] + (snake.right[0] * game.gridsize)) % game.height >= game.width - game.gridsize))
        dir_up = snake.direction == snake.up
        dir_down = snake.direction == snake.down
        dir_left = snake.direction == snake.left
        dir_right = snake.direction == snake.right
        food_up = snake_head[1] < food.position[1]
        food_down = snake_head[1] > food.position[1]
        food_left = snake_head[0] < food.position[0]
        food_right = snake_head[0] > food.position[0]

        state = [danger_up, danger_down, danger_left, danger_right,
                 dir_up, dir_down, dir_left, dir_right,
                 food_up, food_down, food_left, food_right]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0
        return np.asarray(state)
    
    def set_reward(self, snake, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if snake.has_eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            mini_batch = random.sample(memory, batch_size)
        else:
            mini_batch = memory
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 12)))[0])
        target_f = self.model.predict(state.reshape((1, 12)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 12)), target_f, epochs=1, verbose=0)
