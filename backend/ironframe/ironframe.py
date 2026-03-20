import numpy as np

class Tensor:
    __hash__ = None
    
    def __init__(self, data, requires_grad = False):
        self.grad = None
        self.data = np.array(data)
        self.parents = []
        self.requires_grad = requires_grad
        self.backward_fn = None
        self.freezed = False
        
    def backward(self, grad=None):
        if not self.requires_grad and not self.freezed:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        if self.backward_fn is None:
            return

        grads_to_parents = self.backward_fn(grad)

        for parent, parent_grad in zip(self.parents, grads_to_parents):
            parent.backward(parent_grad)
            
    @property            
    def detach(self):
        return Tensor(self.data, requires_grad=False)
            
    @property
    def transpose(self):
        # 1. Forward Pass: Transpose the data
        new_data = self.data.T
        
        # 2. Create the output tensor
        out = Tensor(new_data, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            # 3. Build the graph
            out.parents = [self]
            
            # 4. Define the backward function
            # The gradient flowing back to the input is just the 
            # transpose of the gradient flowing into the output.
            def _backward(grad):
                return [grad.T]
            
            out.backward_fn = _backward
        
        return out
    
    def permute(self, *args):
        # If < 2 dimensions passed.
        if len(args) < 2:
            raise Exception('Error: Tensor.permute requries minimum 2 dimensions.')
        
        axes = args
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
    
        if self.requires_grad:
            out.parents = [self]
    
            def _backward(grad):
                reverse_axes = np.argsort(axes)
                return [np.transpose(grad, reverse_axes)]
    
            out.backward_fn = _backward
    
        return out
    
    def reshape(self, *args):
        # same flexible calling style as transpose
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            new_shape = tuple(args[0])
        else:
            new_shape = args

        original_shape = self.data.shape

        out = Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)

        if self.requires_grad:
            out.parents = [self]
            def _backward(grad):
                return [grad.reshape(original_shape)]
            out.backward_fn = _backward
            
        return out
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return len(self.shape)
    
    def __add__(self, other):
        return add(self, other) 
    
    def __radd__(self, other):
        return add(other, self)
       
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Power must be a non-negative integer for now.")
        if power == 0:
            return Tensor(np.ones_like(self.data), requires_grad=self.requires_grad)
        if power == 1:
            return self
        result = self
        for _ in range(power - 1):
            result = result * self
        return result
    
    def __lt__(self, other):
        other = _totensor(other)
        return Tensor(self.data < other.data, requires_grad=False)

    def __eq__(self, other):
        other = _totensor(other)
        return Tensor(self.data == other.data, requires_grad=False)

    def __ge__(self, other):
        other = _totensor(other)
        return Tensor(self.data >= other.data, requires_grad=False)

    def __gt__(self, other):
        other = _totensor(other)
        return Tensor(self.data > other.data, requires_grad=False)
    
    def __le__(self, other):
        other = _totensor(other)
        return Tensor(self.data <= other.data, requires_grad=False)

    def __ne__(self, other):
        other = _totensor(other)
        return Tensor(self.data != other.data, requires_grad=False)

    def __iadd__(self, other):
        other = _totensor(other)
        self.data = self.data + other.data
        return self

    def __isub__(self, other):
        other = _totensor(other)
        self.data = self.data - other.data
        return self

    def __imul__(self, other):
        other = _totensor(other)
        self.data = self.data * other.data
        return self
    
    def freeze(self):
        detached = self.detach
        detached.freezed = True
        # print("Freezed tensor: ", detached)
        return detached

def _totensor(*args):
    if len(args) < 1:
        raise Exception('Error: _totensor() requries minimum 1 argument')
    elif len(args) == 1:
        t = args[0]
        return t if isinstance(t, Tensor) else Tensor(t, requires_grad=False)
    else:
        t = [t if isinstance(t, Tensor) else Tensor(t, requires_grad=False)
             for t in args]
        return tuple(t)
    
def _revbroadcast(grad_t, target_t):
    while(grad_t.ndim > target_t.ndim):
        grad_t = grad_t.sum(axis = 0)
        
    for dim, (g_dim, t_dim) in enumerate(zip(grad_t.shape, target_t.shape)):
        if t_dim == 1 and g_dim != 1:
            grad_t = grad_t.sum(axis = dim, keep_dims = True)
        
    return grad_t
            
def add(t1, t2):
    t1, t2 = _totensor(t1, t2)
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data + t2.data, requires_grad=requires_grad)    
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            return [_revbroadcast(grad, t1), _revbroadcast(grad, t2)]
        out.backward_fn = backward_fn
    return out

def sub(t1, t2):
    t1, t2 = _totensor(t1, t2)
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data - t2.data, requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            return [_revbroadcast(grad, t1), -_revbroadcast(grad, t2)]
        out.backward_fn = backward_fn
    return out

def mul(t1, t2):
    t1, t2 = _totensor(t1, t2)
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data * t2.data, requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            grad_t1 = _revbroadcast(grad * t2.data, t1)  # scale by t2, then _unrevbroadcast to t1's shape
            grad_t2 = _revbroadcast(grad * t1.data, t2)  # scale by t1, then _unrevbroadcast to t2's shape
            return [grad_t1, grad_t2]
        out.backward_fn = backward_fn
    return out

def div(t1, t2):
    t1, t2 = _totensor(t1, t2)
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data / t2.data, requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            return [
                grad / t2.data,
                -grad * t1.data / (t2.data ** 2)
            ]
        out.backward_fn = backward_fn
    return out

def sum(t, axis = 0, keepdims=True):
    t = _totensor(t)
    out = Tensor(np.sum(t.data, axis=axis, keepdims=keepdims), requires_grad=t.requires_grad)
    if t.requires_grad:
        out.parents = [t]
        def backward_fn(grad):
            return [np.ones_like(t.data) * grad]
        out.backward_fn = backward_fn
    return out

def mean(t, axis = 0, keepdims=True):
    t = _totensor(t)
    n = t.data.shape[axis] if axis is not None else t.data.size
    out = Tensor(np.sum(t.data, axis=axis, keepdims=keepdims)/n, requires_grad=t.requires_grad)
    if t.requires_grad:
        out.parents = [t]
        def backward_fn(grad):            
            return [np.ones_like(t.data) * grad/n]
        out.backward_fn = backward_fn
    return out

def matmul(t1, t2):
    t1, t2 = _totensor(t1, t2)
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(np.matmul(t1.data, t2.data), requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):            
            return [np.matmul(grad, t2.data.T), np.matmul(t1.data.T, grad)]
        out.backward_fn = backward_fn
    return out

def sqrt(t):
    t = _totensor(t)
    out = Tensor(np.sqrt(t.data), requires_grad=t.requires_grad)
    if t.requires_grad:
        out.parents = [t]
        def backward_fn(grad):            
            return [grad / (2 * np.sqrt(t.data))]
        out.backward_fn = backward_fn
    return out
