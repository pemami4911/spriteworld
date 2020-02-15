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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import tasks
from spriteworld.configs.cobra import common
import numpy as np

TERMINATE_DISTANCE = 0.075
NUM_TARGETS = 2 #np.random.choice([2]) 
MODES_SHAPES = {
    'train': distribs.Discrete('shape', ['square', 'circle', 'triangle'], probs=[0.55, 0.25, 0.2]),
    'test': distribs.Discrete('shape', ['triangle', 'circle']),
}
MOTION_STD_DEV = np.array([0,0,0.075, 0.075])
PROPORTIONAL_MOTION_NOISE = 0


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  factors = distribs.Product([
      MODES_SHAPES[mode],
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('scale', [0.2]),
      distribs.Discrete('c0', [0.9, 0.55, 0.27], probs=[0.333,0.334,0.333]),
      distribs.Discrete('c1', [0.6]),
      distribs.Continuous('c2', 0.9, 1.),
  ])
  sprite_gen = sprite_generators.generate_sprites(
      factors, num_sprites=NUM_TARGETS, fix_colors=True)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  task = tasks.FindGoalPosition(terminate_distance=TERMINATE_DISTANCE, sparse_reward=True)

  config = {
      'task': task,
      'action_space': common.noisy_action_space(MOTION_STD_DEV, PROPORTIONAL_MOTION_NOISE, None),
      'renderers': common.renderers(),
      'init_sprites': sprite_gen,
      'max_episode_length': 60,
      'metadata': {
          'name': os.path.basename(__file__),
          'mode': mode
      }
  }
  return config
