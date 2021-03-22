import sys
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from os import makedirs
from os.path import join

import cv2 as cv
import numpy as np
import gym

from utils import rgb_bgr

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


# Key codes
class KeyCode(Enum):
    LEFT_ARROW = 65361
    RIGHT_ARROW = 65363
    UP_ARROW = 65362
    DOWN_ARROW = 65364


DEFAULT_ACTION = 1
ACTION_MAPPING = defaultdict(lambda: DEFAULT_ACTION)
ACTION_MAPPING[KeyCode.LEFT_ARROW.value] = 0
ACTION_MAPPING[KeyCode.RIGHT_ARROW.value] = 2
ACTION_MAPPING[KeyCode.UP_ARROW.value] = 1
ACTION_MAPPING[KeyCode.DOWN_ARROW.value] = 1


class SimpleRandomAgent(object):

    def __init__(self, action_space, num_repeated_actions, action_space_probs=None, different_action=False):
        self.action_space = action_space
        self.num_repeated_actions = num_repeated_actions
        self.action_space_probs = action_space_probs
        self.current_repeat_count = 0
        self.curr_action = None
        self.different_action = different_action

    def get_next_action(self):
        if self.curr_action is None or self.current_repeat_count % self.num_repeated_actions == 0:
            a = np.random.choice(self.action_space, p=self.action_space_probs)
            while self.different_action and self.curr_action == a:
                a = np.random.choice(self.action_space, p=self.action_space_probs)
            self.current_repeat_count = 0
            self.curr_action = a

        self.current_repeat_count += 1
        return self.curr_action


def main():
    # env_id = "CartPole-v0"
    # env_id = "CartPole-v1"
    env_id = "MountainCar-v0"
    # env_id = "Acrobot-v1"
    # env_id = "CubeCrash-v0"
    # env_id = "Freeway-v0"

    save_to_file = True
    save_path = join("keyboard_agent_frames", f"{env_id}_new")
    # save_path = join("keyboard_agent_frames", f"{env_id}")
    # save_path = join("keyboard_agent_frames", f"test_files_{env_id}")
    makedirs(save_path, exist_ok=True)

    env = gym.make(env_id if len(sys.argv) < 2 else sys.argv[1])

    if not hasattr(env.action_space, "n"):
        raise Exception("Keyboard agent only supports discrete action spaces")
    ACTIONS = env.action_space.n
    SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that"s how you
    # can test what skip is still usable.

    def key_press(key, mod):
        global human_agent_action, human_wants_restart, human_sets_pause
        if key == 0xff0d:
            human_wants_restart = True
        if key == 32:
            human_sets_pause = not human_sets_pause
        print(f"Key: {key}")
        a = ACTION_MAPPING[key]
        if a < 0 or a >= ACTIONS:
            human_agent_action = DEFAULT_ACTION
        else:
            human_agent_action = a

    def key_release(key, mod):
        global human_agent_action
        a = ACTION_MAPPING[key]
        if a < 0 or a >= ACTIONS:
            human_agent_action = DEFAULT_ACTION
        else:
            human_agent_action = a

    env.reset()
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    def rollout(env):
        global human_agent_action, human_wants_restart, human_sets_pause
        human_wants_restart = False
        obser = env.reset()
        skip = 0
        total_reward = 0
        total_timesteps = 0
        step_id = 0
        run_id = f"{datetime.now():%d%m%Y_%H%M%S_%f}"

        simple_agent = SimpleRandomAgent([0, 1, 2], np.random.randint(1, 100), different_action=False)

        while True:
            # print("taking action {}".format(human_agent_action))
            # a = human_agent_action
            a = simple_agent.get_next_action()
            total_timesteps += 1

            print(f"Action {a}")
            obser, r, done, info = env.step(a)
            if r != 0:
                # print("reward %0.3f" % r)
                pass
            total_reward += r

            frame = env.render(mode="rgb_array")

            if save_to_file:
                cv.imwrite(join(save_path, f"{run_id}_{a}_{step_id}.jpg"), rgb_bgr(frame))

            step_id += 1
            if frame is None:
                return False
            if done:
                break
            if human_wants_restart:
                break
            while human_sets_pause:
                env.render()
                time.sleep(0.1)
            time.sleep(0.1)
        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

    while True:
        window_still_open = rollout(env)
        print("Restart")
        # if not window_still_open:
        #     break


if __name__ == "__main__":
    main()
