#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class SimpleUUID:
    '''
    A simple monotonically incrementing integer generator.
    '''
    def __init__(self):
        self.current = 0

    def fetch(self):
        curr_id = self.current
        self.current += 1
        return curr_id

if __name__ == "__main__":
    generator = SimpleUUID()
    for _ in range(10):
        print(generator.fetch())