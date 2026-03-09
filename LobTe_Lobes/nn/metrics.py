"""
File: metrics.py
Author: Ariel Hernán Curiale
Github: https://gitlab.com/Curiale
Description:
"""


class Metric:
    """This class defines a metric where the value returned is the mean metric
    value"""

    def __init__(self, name, fx):
        self.name = name
        self.fx = fx
        self.reset_states()

    @property
    def val(self):
        return self._val

    def __call__(self, x, y):
        return self.fx(x, y)

    def update_state(self, x, y):
        # NOTE: Use item to detach the tensor and avoid OOM
        self._val = self.__call__(x, y).item()
        self._state += self._val
        self._n += 1

    def reset_states(self):
        self._val = 0
        self._state = 0
        self._n = 0

    def result(self):
        return self._state / self._n
