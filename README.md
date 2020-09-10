# Reinforcement Learning: Deep Q-Learning 
## Project: Train an AI agent to play snake

## Description
The main goal of this project is to program a small snake game and modify it in order to train an AI bot agent to play the game from scratch. To achieve this goal, I implemented a reinforcement learning algorithm based on the Deep Q-Learning approach. This approach consists of getting the AI agent's state on each move and use positive and negative rewards to teach it how to make the correct moves so that it maximizes the total reward. Since no explicit rules are given to the AI agent, it relies on a neural network to be trained in order to make the moves. Initially, the AI can only make random moves, but as it gets trained we can see it develop a solid strategy and score up to 45 points.

## Install
```bash
git clone git@github.com:jehasbani/snake_reinforcement_learning.git
```

## Run 
To run (or just play, because why wouldn't you?) the game, you have three options:

### First version:
You can use your PyCharm IDE to run snake.py or type in the terminal:
```python
python snake.py
```

### Second version:
You can use your PyCharm IDE to run snake_2.py or type in the terminal:
```python
python snake_2.py
```
This version of the game was modified from the previous version as the first step to get to the training part.

### Reinforcement learning version:
You can use your PyCharm IDE to run snake_RL.py or type in the terminal:
```python
python snake_RL.py
```
You can change the define_parameters() function params dictionary values to view the AI play the game on different scenarios:

#### AI not trained:
```python
params['load_weights'] = False
params['train'] = False
```

#### AI trained:
```python
params['load_weights'] = True
params['train'] = False
```

#### Train AI from scratch:
```python
params['load_weights'] = False
params['train'] = True
```

#### Train AI some more:
```python
params['load_weights'] = True
params['train'] = True
```
