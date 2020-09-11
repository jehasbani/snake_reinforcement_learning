import pygame
import sys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from RL_DQN import Agent
from random import randint
from keras.utils import to_categorical


def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1 / 75
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 150
    params['second_layer_size'] = 150
    params['third_layer_size'] = 150
    params['n_attempts'] = 150
    params['memory_size'] = 2500
    params['batch_size'] = 500
    params['weights_path'] = 'weights/weights.hdf5'
    params['load_weights'] = False
    params['train'] = True
    return params


class Game():
    def __init__(self, game_width, game_height):
        self.width = game_width
        self.height = game_height
        self.gridsize = 20
        self.grid_width = game_width / self.gridsize
        self.grid_height = game_height / self.gridsize
        self.display = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.snake = Snake(self)
        self.food = Food(self)
        self.score = 0

    def draw(self, record):
        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))
        text_score_number = myfont.render(str(self.score), True, (0, 0, 0))
        text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
        self.display.blit(text_score, (45, 440))
        self.display.blit(text_score_number, (120, 440))
        self.display.blit(text_highest, (190, 440))
        self.display.blit(text_highest_number, (350, 440))
        self.display.blit(self.bg, (10, 10))


class Snake():
    def __init__(self, game):
        self.positions = [((game.width / 2), (game.height / 2))]
        self.up = (0, -1)
        self.down = (0, 1)
        self.left = (-1, 0)
        self.right = (1, 0)
        self.direction = random.choice([self.up, self.down, self.left, self.right])
        self.color = (17, 24, 47)
        self.score = 0
        self.length = 1
        self.has_eaten = False
        self.bump_itself = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.step = game.gridsize

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self, game):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * game.gridsize)) % game.width), (cur[1] + (y * game.gridsize)) % game.height)
        if len(self.positions) > 2 and new in self.positions[2:]:
            game.crash = True
            self.bump_itself = True
        elif new[0] < self.step or new[0] >= game.width - self.step or new[1] < self.step or new[1] >= game.height - self.step:
            game.crash = True
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def eat(self, food, game):
        if self.get_head_position() == food.position:
            self.has_eaten = True
            self.length += 1
            game.score += 1
            food.randomize_position(game)
            self.draw(game)
            food.draw(game)
        else:
            self.has_eaten = False

    def handle_action(self, action, food, game):

        if np.array_equal(action, [1, 0, 0, 0]):
            self.turn(self.up)
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.turn(self.down)
        elif np.array_equal(action, [0, 0, 1, 0]):
            self.turn(self.left)
        elif np.array_equal(action, [0, 0, 0, 1]):
            self.turn(self.right)

        self.move(game)
        self.eat(food, game)

    def draw(self, game):
        if not game.crash:
            for p in self.positions:
                game.display.blit(self.image, p)
            update_screen()
        else:
            pygame.time.wait(300)


class Food():
    def __init__(self, game):
        self.position = (0, 0)
        self.randomize_position(game)
        self.image = pygame.image.load('img/food2.png')

    def randomize_position(self, game):
        self.position = (random.randint(1, game.grid_width - 2) * game.gridsize,
                         random.randint(1, game.grid_height - 2) * game.gridsize)
        if self.position in game.snake.positions:
            self.randomize_position(game)

    def draw(self, game):
        game.display.blit(self.image, self.position)
        update_screen()


def update_screen():
    pygame.display.update()
    pygame.event.get()


def initialize_game(game, snake, food, agent, batch_size):
    state_init1 = agent.get_state(game, snake, food)  # [0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
    action = to_categorical(randint(0, 3), num_classes=4)
    snake.handle_action(action, food, game)
    state_init2 = agent.get_state(game, snake, food)
    reward1 = agent.set_reward(snake, game.crash)
    agent.remember(state_init1, action, reward1, state_init2, game.crash)
    agent.replay_new(agent.memory, batch_size)


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def display(snake, food, game, record):
    game.display.fill((255, 255, 255))
    game.draw(get_record(game.score, record))
    snake.draw(game)
    food.draw(game)


def plot_score(array_counter, array_score):
    sns.set(color_codes=True)
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        color="b",
        x_jitter=.1,
        line_kws={'color': 'green'}
    )
    ax.set(xlabel='games', ylabel='score')
    plt.pause(0.05)
    plt.show()


def run(view, speed, params):
    pygame.init()
    clock = pygame.time.Clock()
    agent = Agent(params)
    weights_filepath = params['weights_path']

    if params['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    counter_games = 0
    score_plot = []
    counter_plot = []
    record = 0

    while counter_games < params['n_attempts']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Initialize classes
        game = Game(440, 440)
        snake_agent = game.snake
        food_n = game.food

        # Perform first move
        initialize_game(game, snake_agent, food_n, agent, params['batch_size'])
        if view:
            display(snake_agent, food_n, game, record)
            pygame.time.wait(speed)

        while not game.crash:
            clock.tick(10)
            if not params['train']:
                agent.epsilon = 0
            else:
                # agent.epsilon is set to give randomness to actions
                agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

            # get old state
            state_old = agent.get_state(game, snake_agent, food_n)

            # perform random actions based on agent.epsilon, or choose the action
            if randint(0, 1) < agent.epsilon:
                action = to_categorical(randint(0, 3), num_classes=4)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 12)))
                action = to_categorical(np.argmax(prediction[0]), num_classes=4)

            # perform new move and get new state
            snake_agent.handle_action(action, food_n, game)
            record = get_record(game.score, record)
            state_new = agent.get_state(game, snake_agent, food_n)

            if view:
                display(snake_agent, food_n, game, record)
                update_screen()

            # set reward for the new state
            reward = agent.set_reward(snake_agent, game.crash)

            if params['train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, action, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, action, reward, state_new, game.crash)

        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])

        counter_games += 1
        print(f'Game {counter_games}      Score: {game.score}')
        score_plot.append(game.score)
        counter_plot.append(counter_games)
        if params['train']:
            if game.score == record:
                agent.model.save_weights(f'weights/weights.hdf5')

    plot_score(counter_plot, score_plot)


if __name__ == '__main__':
    pygame.font.init()
    params = define_parameters()
    view_process = True
    speed = 50
    run(view_process, speed, params)
