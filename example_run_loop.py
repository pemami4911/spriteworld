# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Template for running an agent on Spriteworld tasks.

This script runs an agent on a Spriteworld task. The agent takes random actions
and does not learn, so this serves only as an example of how to run an agent in
the environment, logging task success and mean rewards.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import app
from absl import flags
from absl import logging
import numpy as np
from six.moves import range

from spriteworld import environment
from spriteworld import renderers

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 30, 'Number of training episodes.')
flags.DEFINE_string('config',
                    'spriteworld.configs.mon.goal_finding_planning',
                    'Module name of task config to use.')
flags.DEFINE_string('mode', 'train', 'Task mode, "train" or "test"]')


class RandomAgent(object):
  """Agent that takes random actions."""

  def __init__(self, env):
    """Construct random agent."""
    self._env = env

  def step(self, timestep):
    # observation is a dictionary with renderer outputs to be used for training
    observation = timestep.observation
    del observation
    del timestep
    
    pos = self._env.sample_contained_position()
    action = self._env.action_space.sample()
    action[:2] = pos
    return action


def main(argv):
  del argv
  config = importlib.import_module(FLAGS.config)
  config = config.get_config(FLAGS.mode)
  config['renderers']['success'] = renderers.Success()  # Used for logging
  env = environment.Environment(**config)
  agent = RandomAgent(env)

  episode_frames = np.zeros((20,64,64,3))
  # Loop over episodes, logging success and mean reward per episode
  for episode in range(FLAGS.num_episodes):
    timestep = env.reset()
    t = 0
    rewards = []
    while not timestep.last():
      episode_frames[t] = timestep.observation['image']
      action = agent.step(timestep)
      timestep = env.step(action)
      rewards.append(timestep.reward)
      t += 1
    logging.info('Episode %d: Success = %r, Reward = %s.', episode,
                 timestep.observation['success'], np.nanmax(rewards))
  np.save('random_actions', episode_frames)

if __name__ == '__main__':
  app.run(main)
