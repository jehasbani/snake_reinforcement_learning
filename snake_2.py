import pygame
import sys
import random
import numpy as np
import seaborn as sns
from random import randint


class Game():
    def __init__(self, game_width, game_height):
        self.width = game_width
        self.height = game_height
        self.gridsize = 20
        self.grid_width = game_width / self.gridsize
        self.grid_height = game_height / self.gridsize
        self.display = pygame.display.set_mode((game_width, game_height+60))
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
        # new = (((cur[0] + (x * self.step)) % game.width), (cur[1] + (y * self.step)) % game.height)
        new = (((cur[0] + (x * game.gridsize)) % game.width), (cur[1] + (y * game.gridsize)) % game.height)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset(game)
        elif new[0] < self.step or new[0] >= game.width - self.step or new[1] < self.step or new[1] >= game.height - self.step:
            self.reset(game)
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

    def handle_action(self, food, game):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.turn(self.up)
                elif event.key == pygame.K_DOWN:
                    self.turn(self.down)
                elif event.key == pygame.K_LEFT:
                    self.turn(self.left)
                elif event.key == pygame.K_RIGHT:
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

    def reset(self, game):
        self.length = 1
        self.positions = [((game.width / 2), (game.height / 2))]
        self.direction = random.choice([self.up, self.down, self.left, self.right])
        game.score = 0


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


def display(snake, food, game, score, record):
    game.display.fill((255, 255, 255))
    game.draw(get_record(score, record))
    snake.draw(game)
    food.draw(game)


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def main():
    pygame.init()

    clock = pygame.time.Clock()
    game = Game(440, 440)
    snake = game.snake
    food = game.food
    record = 0

    while (True):
        clock.tick(10)
        snake.handle_action(food, game)
        record = get_record(game.score, record)
        display(snake, food, game, game.score, record)
        update_screen()


main()
