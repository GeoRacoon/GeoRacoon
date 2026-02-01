""" Provides a simple context manager class for timing code execution and
recording intermediate durations (laps).
"""

from __future__ import annotations
from time import perf_counter


class TimedTask:
    """
    A context manager for measuring elapsed time of code blocks.

    This class allows timing of an entire block of code and
    recording intermediate checkpoints ("laps") within it.

    Attributes
    ----------
    labs : (list[float])
        Stores elapsed times for each recorded lap.

    Examples
    ---------
        >>> with TimedTask() as t:
        ...     do_work()
        ...     t.new_lab()
        ...     do_more_work()
        ...
        >>> t.get_duration()
        2.3512  # total seconds elapsed
        >>> t.labs
        [0.7543, 1.5969]  # lap times
    """

    def __init__(self):
        self.labs = []

    def __enter__(self) -> TimedTask:
        """
        Start the timer upon entering the context.

        Returns
        -------
        TimedTask
             The timer instance itself.
        """
        self.now = perf_counter()
        self.start = self.now
        self.stop = 0.0
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Stop the timer upon exiting the context.

        Records a final lap and stores the total elapsed time
        between start and stop.
        """
        self.new_lab()
        self.stop = self.now
        self.dt = self.stop - self.start

    def get_duration(self) -> float:
        """
        Get the total duration measured between start and stop.

        Returns
        -------
        float
            Total elapsed time in seconds.
        """
        return self.stop - self.start

    def new_lab(self):
        """
        Record a new lap (checkpoint).

        Each call appends the time elapsed since the previous
        lap (or since the start if it's the first lap).
        """
        self.now = perf_counter()
        try:
            self.labs.append(self.now - self.labs[-1])
        except IndexError:
            self.labs.append(self.now - self.start)