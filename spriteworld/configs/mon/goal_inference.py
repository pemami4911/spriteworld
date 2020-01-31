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

NUM_TARGETS = 2
MODES_SHAPES = {
    'train': distribs.Discrete('shape', ['square', 'circle', 'triangle'], probs=[0.6, 0.15, 0.25]),
    'test': distribs.Discrete('shape', ['triangle', 'circle']),
}
GOAL=True


def get_config(mode='train'):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  non_goal_factors = distribs.Product([
        distribs.Mixture([distribs.Continuous('x', 0.1, 0.4),distribs.Continuous('x', 0.6,0.9)]),
        distribs.Mixture([distribs.Continuous('y', 0.1, 0.4),distribs.Continuous('y', 0.6,0.9)]),
      ])
  goal_factors = distribs.Product([
      distribs.Continuous('x', 0.45,0.55),
      distribs.Continuous('y', 0.45,0.55)
    ])

  factors = distribs.Product([
      MODES_SHAPES[mode],
      distribs.Discrete('scale', [0.13]),
      distribs.Discrete('c0', [0.9, 0.55, 0.27], probs=[0.6,0.3,0.1]),
      distribs.Discrete('c1', [0.6]),
      distribs.Continuous('c2', 0.9, 1.),
  ])

  if GOAL:
      factors = distribs.Product((goal_factors, factors))
  else:
      factors = distribs.Product((non_goal_factors, factors))

  sprite_gen = sprite_generators.generate_sprites(
      factors, num_sprites=NUM_TARGETS)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)
  task = tasks.NoReward()

  config = {
      'task': task,
      'action_space': common.action_space(),
      'renderers': common.renderers(),
      'init_sprites': sprite_gen,
      'max_episode_length': 1,
      'metadata': {
          'name': os.path.basename(__file__),
          'mode': mode
      }
  }
  return config
