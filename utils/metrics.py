import torch

from allennlp.training.metrics.metric import Metric


class ScalarMetric(Metric):

    def __init__(self):
        self._sum = 0
        self._count = 0

    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._sum += value
        self._count += 1

    def get_metric(self, reset: bool = False):
        avg = self._sum / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return avg

    def reset(self):
        self._sum = 0
        self._count = 0
