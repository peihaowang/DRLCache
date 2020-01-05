
import sys, os

class Cache(object):
    def __init__(self, progs, cache_size, interval=0):
        # Load programs and simulate the disk accessed
        # Initialize and fill cache
        pass

    def step(self, action):
        # Step through the action
        # Evict cache
        # Wait for the next miss
        # Return observation, reward
        pass

    def hasDone(self):
        # Has done?
        pass

    def display(self):
        # Display the current cache state
        pass

    def hit_rate(self):
        # Return hit rate
        pass
