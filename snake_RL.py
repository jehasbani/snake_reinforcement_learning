import pygame
import sys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from RL_DQN import Agent
from random import randint
from keras.utils import to_categorical


def define_parameters():
    params = dict()
    params['epsilon_decay_linear'] = 1 / 75
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 150  # neurons in the first layer
    params['second_layer_size'] = 150  # neurons in the second layer
    params['third_layer_size'] = 150  # neurons in the third layer
    params['episodes'] = 150
    params['memory_size'] = 2500
    params['batch_size'] = 500
    params['weights_path'] = 'weights/weights.hdf5'
    params['load_weights'] = True
    params['train'] = False
    return params


class Game():
    def __init__(self, game_width, game_height):
        self.width = game_width
        self.height = game_height
        self.display = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.snake = Snake(self)
        self.food = Food()
        self.score = 0


class Snake():
    def __init__(self, game):
        self.positions = [((game.width / 2), (game.height / 2))]
        self.direction = random.choice([game.up, game.down, game.left, game.right])
        self.color = (17, 24, 47)
        self.score = 0
        self.length = 1
        self.has_eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.step = 20
        self.up = (0, -1)
        self.down = (0, 1)
        self.left = (-1, 0)
        self.right = (1, 0)

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
        new = (((cur[0] + (x * self.step)) % game.width), (cur[1] + (y * self.step)) % game.height)
        if len(self.positions) > 2 and new in self.positions[2:]:
            game.crash = True
        elif new[0] < self.step or new[0] > game.width - self.step or new[1] < self.step or new[1] > game.height - self.step:
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
            food.randomize_position()
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
                game.gameDisplay.blit(self.image, p)
            update_screen()
        else:
            pygame.time.wait(300)


class Food():
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()
        self.image = pygame.image.load('img/food2.png')

    def randomize_position(self, game):
        self.position = (random.randint(0, game.width - 1), random.randint(0, game.height - 1))

    def draw(self, game):
        game.gameDisplay.blit(self.image, self.position)
        update_screen()


def update_screen():
    pygame.display.update()


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, 440))
    game.gameDisplay.blit(text_score_number, (120, 440))
    game.gameDisplay.blit(text_highest, (190, 440))
    game.gameDisplay.blit(text_highest_number, (350, 440))
    game.gameDisplay.blit(game.bg, (10, 10))


def display(snake, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    snake.display_snake(snake.position[-1][0], snake.position[-1][1], snake.food, game)
    food.draw(food.x_food, food.y_food, game)


def run(view, speed, params):
    pygame.init()


if __name__ == '__main__':
    run()
