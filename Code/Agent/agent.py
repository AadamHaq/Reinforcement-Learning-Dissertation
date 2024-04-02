import torch #pytorch
import random
import numpy as np #numpy
from collections import deque #data structure to store memory
import os

from game_no_render import SnakeGameRL, Direction, Point # Importing the game
from model import Linear_QNet, QTrainer # Importing the NN
#from helper import plot # Importing the plotter
from helper2 import plot, save_mean_score_plot # Importing the plotter


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #learning rate


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.85  # discount rate, must be less than 1. Usually around 0.8 or 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if memory is full
        self.model = Linear_QNet(11, 256, 3) # 11 input states, 256 hidden and 3 output (directions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        

    def get_state(self, game):

        # This model will have 11 states. It will detect if there are dangers in front and to the left of the right of the snake. It then detects what direction the snake is going in (only one will be correct). Finally it detects what direction the fruit is from the snake (combination of the four).

        # Front of the snake
        head = game.snake[0]

        # Points next to the head as the snake blocks are 20 pixels long
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Check current game direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight ahead
            # i.e.if we are heading right and there is danger on the right hand side. Only one of these four lines will give an answer as we can only be heading in one of these directions at a time
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger to the right - same but anti-clockwise rotation
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger to the left - same but clockwise rotation
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Direction, only one of them will be correct
            dir_l, 
            dir_r,
            dir_u,
            dir_d,

            # Food locations are simpler
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y  # Food down
        ]
        # Converts true or false boolians to a 0 or a 1 and returns our state
        return np.array(state, dtype=int) 
        

    def remember(self, state, action, reward, next_state, done):
        # memory is a deque
        self.memory.append((state, action, reward, next_state, done))  # Stored as one big tuple. popleft if MAX_MEMORY is reached. This means furthest away in time is forgotten

    def train_long_memory(self):
        # Check if there are over 1000 samples in our memory
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # random list of 1000 tuples. Our memory was stored as one large tuple for each game. 

        # If we do not have 1000 samples in the memory, we take the whole memory
        else:
            mini_sample = self.memory

        # The * in our zip will unpack the tuples so we can collate all the similar ones together. Then we can put this into our trainer. This could've been done in a for loop, but this is easier and faster for pytorch
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Trains for only one game step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # We want some random moves. This is for the exploration/exploitation trade off
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        # After 80 moves, it will be impossible for there to be a random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            # This random number (either 0,1 or 2) will determine if the next move is straight, left or right
            final_move[move] = 1 

        # We want a move based on our model
        else:
            # Convert state to a tensor
            state0 = torch.tensor(state, dtype=torch.float) 

            # Predict our next step
            prediction = self.model(state0)

            # The prediction will return an array based off the weights. E.g. [5.0, 8.2, 0.4] we then want to move in the direction with the best predictor, so in this case it would be [0,1,0]
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameRL()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get results
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game) # with our new game

        # train sm (1 step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done) # Uses next state from old

        # Need to remember these
        agent.remember(state_old, final_move, reward, state_new, done) 

        # What to do if there is a game over (done)
        if done: 
            # train long memory / 'experience replay' and plotting
            # Trains on all previous moves and games played
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score # Get high score
                agent.model.save()

            # Print useful information
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # Call plot() to display and save both plots
            plot(plot_scores, plot_mean_scores)

            # Call save_mean_score_plot() separately to save only the mean scores plot
            save_mean_score_plot(plot_mean_scores)

if not os.path.exists('frames'):
    os.makedirs('frames')

if __name__ == '__main__':
    # Train the agent
    train()
