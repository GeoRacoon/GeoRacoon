"""
Provides a simple class that allows the timing of function calls.

"""
from __future__ import annotations
from time import perf_counter

class TimedTask:
    def __init__(self,):
        self.labs = []

    def __enter__(self):
        self.now = perf_counter()
        self.start = self.now
        self.stop = 0.0
        return self

    def get_duration(self, ):
        return self.stop - self.start

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.new_lab()
        self.stop = self.now
        self.dt = self.stop - self.start

    def new_lab(self,):
        self.now = perf_counter()
        try:
            self.labs.append(self.now - self.labs[-1])
        except IndexError:
            self.labs.append(self.now - self.start)
