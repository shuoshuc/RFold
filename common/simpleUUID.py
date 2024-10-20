class SimpleUUID:
    '''
    A simple monotonically incrementing integer generator.
    '''

    def __init__(self):
        self.current = 0

    def fetch(self) -> int:
        curr_id = self.current
        self.current += 1
        return curr_id
