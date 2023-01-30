import argparse
import logging
import time
import random

import numpy as np
import pygame as pg

from agent import Agent
from environment import World

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run(args):
    logger.info("Creating new agent...")

    logger.info("Creating new environment...")

    # training loop
    logger.info(f"Starting training loop with {args.episodes} episodes")
    size = args.size
    env = World(size=size, picker_position=(0, 0), mushroom_position=(0, 0))
    number_of_states = env.get_number_of_states()
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon

    agent = Agent(number_of_states, alpha, gamma, epsilon)

    all_steps = []
    pg.init()

    white = 255, 240, 200  # position of picker
    black = 20, 20, 40  # background
    red = 255, 0, 0  # position of mashroom
    blue = 0, 0, 255  # action 4 - pick the mashroom
    green = 0, 255, 0  # star picker position
    grey = 120, 120, 120  # where picker was
    grey_blue = 120, 120, 255  # where picker was trying to pick up mashroom
    multi = 30  # multiplication of field size
    divis = 200  # episodes to display
    for episode in range(args.episodes):

        if episode % divis == 0:
            screen = pg.display.set_mode([multi * size, multi * size])
            screen.fill(black)
            pg.display.set_caption(f'Episode {episode}')
            pg.display.update()
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        # x = 8
        # y = 7
        env = World(size=10, picker_position=(0, 0), mushroom_position=(x, y))

        done = False
        steps = 0
        pick_pos = []
        while not done:
            state = env.get_state()
            if episode % divis == 0:
                pg.event.pump()
                pg.draw.rect(screen, red, (multi * x, multi * y, multi, multi))
                pg.draw.rect(screen, grey,
                             (multi * env.picker_position[0], multi * env.picker_position[1], multi, multi))
                for pos in pick_pos:
                    pg.draw.rect(screen, grey_blue,
                                 (multi * pos[0], multi * pos[1], multi, multi))

                pg.draw.rect(screen, green, (0, 0, multi, multi))
                pg.display.update()

            action = agent.act(state)
            reward, done = env.step(action)
            new_state = env.get_state()
            agent.update(state, new_state, action, reward)

            if episode % divis == 0:
                if action == 4:
                    color = blue
                    pick_pos.append(env.picker_position)

                else:
                    color = white
                pg.draw.rect(screen, color,
                             (multi * env.picker_position[0], multi * env.picker_position[1], multi, multi))
                pg.display.update()
                time.sleep(0.02)
            steps += 1

        print(steps)
        all_steps.append(steps)
        print(f'next episode {episode + 1}')
        if episode % divis == 0:
            time.sleep(5)
    time.sleep(300)
    logger.info(f'average steps per run {np.array(all_steps).sum() / args.episodes}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000, help="How many episodes to perform")
    parser.add_argument('--max-steps', type=int, default=1000, help="How many time steps to run within one episode")
    parser.add_argument('--alpha', type=int, default=0.1, help="Learning rate")
    parser.add_argument('--gamma', type=int, default=0.5, help="Discount factor")
    parser.add_argument('--epsilon', type=int, default=0.1, help="Random action threshold")
    parser.add_argument('--size', type=int, default=10, help="World size")
    arguments = parser.parse_args()

    logger.info("Arguments:")
    logger.info(arguments)

    run(arguments)
