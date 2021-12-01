""" Contains the Episodes for Navigation. """
import random

import torch
import numpy as np
from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY
from datasets.constants import DONE
from datasets.environment import Environment

from utils.net_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.net_util import gpuify
from .episode import Episode
import scipy.io as scio


class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.last_state = None
        self.scene_states = []
        self.intersection = []
        self.done_dis2goal = None
        self.atts = scio.loadmat('./data/attributes/40_attribute.mat')
        self.ids = []
        if args.eval:
            random.seed(args.seed)

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def current_att_feature(self):
        return self.environment.current_att_feature

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int):

        action = self.actions_list[action_as_int]

        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful = self.judge(action)
        return reward, terminal, action_was_successful

    def jacard(self, v1, v2):
        v1 = np.squeeze(v1)
        v2 = np.squeeze(v2)
        sim = 0
        for i in range(len(v1)):
            if max(v1[i], v2[i]) == 0:
                sim += 1
            else:
                sim += min(v1[i], v2[i]) / max(v1[i], v2[i])
        sim = float(sim) / len(v1)
        return sim

    def att_reward(self, target, atts, ids):
        tar_att = atts[target]
        max_sim = None
        for id_ in ids:
            if self.environment.object_is_visible(id_):
                find = id_.split("|")[0]
                find_att = atts[find]
                sim = self.jacard(tar_att, find_att)
                if max_sim is not None:
                    if sim > max_sim:
                        max_sim = sim
                else:
                    max_sim = sim
        if max_sim is None:
            return 0
        else:
            return 5 * max_sim

    def state2goal(self,state,task_data):
        sx= state.x
        sz= state.z
        ds = []
        for id in task_data:
            gx = float(id.split('|')[1])
            gz = float(id.split('|')[3])
            d = np.sqrt((sx-gx)**2+(sz-gz)**2)
            ds.append(d)
        return min(ds)

    def distance_reward(self):
        d_last = self.state2goal(self.last_state, self.task_data)
        d_now = self.state2goal(self.environment.controller.state, self.task_data)
        dis_reward = max((d_last - d_now)*0.1, 0)
        return dis_reward

    def judge(self, action):
        """ Judge the last event. """
        reward = STEP_PENALTY
        if self.last_state is None:
            dis_R = 0
        else:
            dis_R = self.distance_reward()
        reward = reward + dis_R
        self.last_state = self.environment.controller.state

        self.done_dis2goal = max(self.state2goal(self.environment.controller.state, self.task_data) - 1, 0)
        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
        else:
            self.scene_states.append(self.environment.controller.state)

        done = False

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    break
            if done:
                reward = self.att_reward(self.target_object, self.atts, self.ids)
                # print(reward)
            else:
                reward = 0.1 * self.att_reward(self.target_object, self.atts, self.ids)
        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def _new_episode(
        self, args, scenes, possible_targets, targets=None, keep_obj=False, glove=None
    ):
        """ New navigation episode. """
        scene = random.choice(scenes)

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        # Randomize the start location.
        self._env.randomize_agent_location()
        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]
        self.intersection = intersection

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        for id_ in objects:
            type_ = id_.split("|")[0]
            if type_ in self.intersection:
                self.ids.append(id_)

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

        self.glove_embedding = None
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[goal_object_type][:], self.gpu_id
        )

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        keep_obj=False,
        glove=None,
    ):
        self.last_state = None
        self.done_dis2goal = None
        self.done_count = 0
        self.ids = []
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None
        self._new_episode(args, scenes, possible_targets, targets, keep_obj, glove)
