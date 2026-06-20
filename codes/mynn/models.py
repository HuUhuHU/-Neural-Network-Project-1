from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []

        if size_list is not None and act_func is not None:
            self._build_layers(size_list, act_func, lambda_list)

    def _build_layers(self, size_list, act_func, lambda_list=None):
        self.layers = []
        for i in range(len(size_list) - 1):
            layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[i]
            if act_func == 'Logistic':
                raise NotImplementedError
            elif act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(size_list) - 2:
                self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self._build_layers(self.size_list, self.act_func)
        param_idx = 2
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[param_idx]['W']
                layer.b = param_list[param_idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[param_idx]['weight_decay']
                layer.weight_decay_lambda = param_list[param_idx]['lambda']
                param_idx += 1
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A simple CNN for MNIST: 3 conv blocks + 2 fully-connected layers.
    Input can be flat [N, 784] or image [N, 1, 28, 28].
    """
    def __init__(self):
        self.layers = [
            conv2D(1, 8, 3, stride=1, padding=1),
            ReLU(),
            conv2D(8, 16, 3, stride=2, padding=1),
            ReLU(),
            conv2D(16, 32, 3, stride=2, padding=1),
            ReLU(),
            Flatten(),
            Linear(32 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10),
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, 28, 28)
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        param_idx = 0
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[param_idx]['W']
                if 'b' in param_list[param_idx]:
                    layer.b = param_list[param_idx]['b']
                layer.params['W'] = layer.W
                if hasattr(layer, 'b'):
                    layer.params['b'] = layer.b
                layer.weight_decay = param_list[param_idx].get('weight_decay', False)
                layer.weight_decay_lambda = param_list[param_idx].get('lambda', 1e-8)
                param_idx += 1
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                entry = {
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                }
                param_list.append(entry)
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
