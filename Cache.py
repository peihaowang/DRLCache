
import sys, os
import numpy as np
import pandas as pd

class Cache(object):

    def __init__(self, progs, cache_size, allow_skip=True, interval=0, boot=False):
        # Load programs and simulate the disk accessed
        # Initialize and fill cache
        self.total_count = 0
        self.miss_count = 0
        self.evict_count = 0
        self.last_action = -1

        self.requests = []
        self.operations = []
        for prog in progs:
            df = pd.read_csv(prog, header=0)
            if not boot: df = df.loc[df['boot/exec'] == 1, :]
            self.requests += list(df['blocksector'])
            self.operations += list(df['read/write'])
        self.cur_index = 0

        if len(self.requests) <= cache_size:
            print("The count of requests are too small. Try larger one.")

        self.cache_size = cache_size
        self.slots = [-1] * self.cache_size
        self.used_times = [-1] * self.cache_size
        self.access_bits = [False] * self.cache_size
        self.dirty_bits = [False] * self.cache_size

    # Display the current cache state
    def display(self):
        print(self.slots)

    # Return miss rate
    def miss_rate(self):
        return self.miss_count / self.total_count

    def reset(self):
        self.slots = self.requests[:self.cache_size]
        self.used_times = list(range(self.cache_size))
        self.access_bits = [True] * self.cache_size
        self.dirty_bits = [self.operations[i] == 1 for i in range(self.cache_size)]
        self.cur_index = self.cache_size
        return self._get_observation()

    # Has program finished?
    def hasDone(self):
        return self.cur_index == len(self.requests)

    def step(self, action):
        assert 0 <= action and action <= len(self.slots)
        if not allow_skip: assert action == 0

        # Evict slot of (aciton - 1). action == 0 means skipping eviction.
        if action != 0:
            out_resource = self.slots[action - 1]
            in_resource = self.requests[self.cur_index]
            slot_id = action - 1
            self.slots[slot_id] = in_resource
            self.hit_cache(slot_id)
            self.evict_count += 1
        else:
            skip_resource = self.requests[self.cur_index]

        last_index = self.cur_index

        # Proceed kernel and resource accesses.
        self.cur_index += 1
        while self.cur_index < len(self.requests):
            request = self.requests[self.cur_index]
            self.total_count += 1
            if request not in self.slots:
                self.miss_count += 1
                break
            else:
                slot_id = self.slots.index(request)
                self.hit_cache(slot_id)
            self.cur_index += 1

        # Get observatio.
        observation = self._get_observation()

        # Compute reward: R = hit reward + miss penalty
        reward = 0.0

        # Total count of hit since last decision epoch
        hit_count = self.cur_index - last_index - 1
        reward += hit_count

        miss_resource = self.requests[self.cur_index]
        # If evction happens at last decision epoch
        if action != 0:
            # Compute the swap-in reward
            past_requests = self.requests[last_index + 1 : self.cur_index]
            reward += 10 * past_requests.count(in_resource)
            # Compute the swap-out penalty
            if miss_resource == out_resource:
                reward -= 100 / (hit_count + 1) + 10
        # Else no evction happens at last decision epoch 
        else:
            # Compute the penalty of skipping eviction
            if miss_resource == skip_resource:
                reward -= 100 / (hit_count + 1) + 10

        return observation, reward

    def hit_cache(self, slot_id):
        # Set access bit
        self.access_bits[slot_id] = True
        # If the operation is write
        if self.operations[self.cur_index] == 1:
            # Set dirty bit
            self.dirty_bits[slot_id] = True
        # Record last used time
        self.used_times[slot_id] = self.cur_index

    # The number of requests on rc_id among last term requests
    def _elapsed_requests_(self, term, rc_id):
        start = self.cur_index - term + 1
        end = self.cur_index+1
        if start < 0: start = 0
        return self.requests[start : end].count(rc_id)

    # Return the observation features for reinforcement agent
    def _get_observation(self):
        # Elasped terms - short, middle and long
        terms = [10, 100, 1000]

        # [Freq, F1, F2, ..., Fc] where Fi = [Rs, Rm, Rl]
        # i.e. the request times in short/middle/long term for each
        # cached resource and the currently requested resource.
        features = np.concatenate([
            np.array([self._elapsed_requests_(t, self.requests[self.cur_index]) for t in terms])
            , np.array([self._elapsed_requests_(t, rc) for rc in self.slots for t in terms])
        ], axis=0)

        return dict(features=features
            , cache_state=self.slots.copy()
            , last_used_times=self.used_times.copy()
            , access_bits=self.access_bits.copy()
            , dirty_bits=self.dirty_bits.copy()
        )
