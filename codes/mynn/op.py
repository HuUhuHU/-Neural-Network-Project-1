from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim)) * 0.01
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        """
        batch_size = self.input.shape[0]
        self.grads['W'] = self.input.T @ grad / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Implemented with im2col for clarity and reasonable speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self._cols = None
        self._out_shape = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _pad_input(self, X):
        if self.padding > 0:
            return np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        return X

    def _im2col(self, X, out_h, out_w):
        batch, in_c, _, _ = X.shape
        k = self.kernel_size
        cols = np.zeros((batch, in_c * k * k, out_h * out_w), dtype=X.dtype)
        col_idx = 0
        for i in range(out_h):
            h_start = i * self.stride
            for j in range(out_w):
                w_start = j * self.stride
                patch = X[:, :, h_start:h_start + k, w_start:w_start + k]
                cols[:, :, col_idx] = patch.reshape(batch, -1)
                col_idx += 1
        return cols

    def _col2im(self, cols, input_shape, out_h, out_w):
        batch, in_c, H, W = input_shape
        k = self.kernel_size
        dX = np.zeros((batch, in_c, H, W), dtype=cols.dtype)
        col_idx = 0
        for i in range(out_h):
            h_start = i * self.stride
            for j in range(out_w):
                w_start = j * self.stride
                patch = cols[:, :, col_idx].reshape(batch, in_c, k, k)
                dX[:, :, h_start:h_start + k, w_start:w_start + k] += patch
                col_idx += 1
        return dX

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        """
        X = self._pad_input(X)
        self.input = X
        batch, _, H, W = X.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1
        self._out_shape = (out_h, out_w)

        cols = self._im2col(X, out_h, out_w)
        self._cols = cols
        W_col = self.W.reshape(self.out_channels, -1)
        out = np.einsum('bik,oi->bok', cols, W_col) + self.b.reshape(1, self.out_channels, 1)
        return out.reshape(batch, self.out_channels, out_h, out_w)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch = grads.shape[0]
        out_h, out_w = self._out_shape
        grads_col = grads.reshape(batch, self.out_channels, -1)
        W_col = self.W.reshape(self.out_channels, -1)

        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        dW_col = np.einsum('bok,bik->oi', grads_col, self._cols) / batch
        self.grads['W'] = dW_col.reshape(self.W.shape)

        dcols = np.einsum('bok,oi->bik', grads_col, W_col)
        dX = self._col2im(dcols, self.input.shape, out_h, out_w)

        if self.padding > 0:
            p = self.padding
            dX = dX[:, :, p:-p, p:-p]

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        """
        self.predicts = predicts
        self.labels = labels.astype(int)

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        batch_size = labels.shape[0]
        log_probs = -np.log(self.probs[np.arange(batch_size), self.labels] + 1e-12)
        loss = np.mean(log_probs)
        return loss
    
    def backward(self):
        batch_size = self.labels.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        self.grads /= batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
