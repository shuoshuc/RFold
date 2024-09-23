#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class SimpleUUID:
    '''
    A simple monotonically incrementing integer generator.
    '''
    def __init__(self):
        self.current = 0

    def fetch(self):
        self.current += 1
        return self.current

if __name__ == "__main__":
    generator = SimpleUUID()
    for _ in range(10):
        print(generator.fetch())