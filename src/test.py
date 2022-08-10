""" Test Agent """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tf info, warning and error messages are not printed
import gym
import logging
from logging import config
import numpy as np
from agent import DQNAgent
import utils
import time

config = utils.load_config("config.yaml")
logging_config = utils.load_logging_config("logging.yaml")

env = gym.make(config["environment"]["name"])
agent = DQNAgent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=config["agent"]["learning_rate"],
                 discount_factor=config["agent"]["discount_factor"], eps=config["agent"]["eps"],
                 eps_dec=config["agent"]["eps_dec"], eps_min=config["agent"]["eps_min"],
                 batch_size=config["agent"]["batch_size"], replace=config["agent"]["replace_target_network_cntr"],
                 use_target_network=config["network"]["use_target_network"],
                 mem_size=config["replay_buffer"]["mem_size"], algo="dqn", env_name=config["environment"]["name"])

start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

# lists for storing data
episode_list = []
score_list = []
avg_score_list = []
epsilon_list = []
best_score = -np.inf


def test():
    logger.info("Start testing")
    for i in range(config["test"]["episodes"]):
        agent.load_models()
        agent.epsilon = 0.0
        done = False
        score = 0
        observation = env.reset()
        if config["render"]:
            env.render()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_

        score_list.append(score)
        avg_score = np.mean(score_list[-100:])
        avg_score_list.append(avg_score)

        logger.info('episode: {}, score: {}, avg_score: {}, epsilon: {}'.format(i, "%.2f" % score, "%.2f" % avg_score,
                                                                                "%.2f" % agent.epsilon))

        logger.info("Finish testing")


if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("test")
    test()
