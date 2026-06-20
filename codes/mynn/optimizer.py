from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

    def _iter_layers(self):
        if hasattr(self.model, 'layers'):
            return self.model.layers
        return []


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self._iter_layers():
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
        for layer in self._iter_layers():
            if layer.optimizable:
                for key in layer.params.keys():
                    self.velocity[id(layer), key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for layer in self._iter_layers():
            if layer.optimizable:
                for key in layer.params.keys():
                    vid = (id(layer), key)
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.velocity[vid] = self.mu * self.velocity[vid] - self.init_lr * layer.grads[key]
                    layer.params[key] += self.velocity[vid]
