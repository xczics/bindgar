from typing import List

class CyclicList:
    def __init__(self, data: List):
        self.data = data
        self.length = len(data)
    def __getitem__(self, index: int):
        return self.data[index % self.length]
    def __len__(self):
        return self.length
    def __iter__(self):
        import itertools
        return itertools.cycle(self.data)
    
default_colors = CyclicList([
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
])# avoid out-of-index error when accessing colors