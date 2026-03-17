import numpy as np

class Tensor:
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
            
    def detach(self):
        return Tensor(self.data, requires_grad=False)
            
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
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return add(self, other) 
    
    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return add(other, self)
       
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return sub(self, other)
    
    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return sub(other, self)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return mul(self, other)

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return mul(other, self)
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return div(self, other)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return div(other, self)
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def __neg__(self):
        return mul(self, Tensor(-1.0, requires_grad=False))
    
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
    
    def __gt__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return Tensor(self.data > other.data, requires_grad=False)

    def __lt__(self, other):
        ...

    def __eq__(self, other):
        ...

    def __ge__(self, other):
        ...

    def __le__(self, other):
        ...
    
    def freeze(self):
        detached = self.detach()
        detached.freezed = True
        # print("Freezed tensor: ", detached)
        return detached

def add(t1, t2):
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data + t2.data, requires_grad=requires_grad)    
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            return [grad, grad]
        out.backward_fn = backward_fn
    return out

def sub(t1, t2):
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data - t2.data, requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            return [grad, -grad]
        out.backward_fn = backward_fn
    return out

def mul(t1, t2):
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(t1.data * t2.data, requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):
            return [
                grad * t2.data,
                grad * t1.data
            ]
        out.backward_fn = backward_fn
    return out

def div(t1, t2):
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
    out = Tensor(np.sum(t.data, axis=axis, keepdims=keepdims), requires_grad=t.requires_grad)
    if t.requires_grad:
        out.parents = [t]
        def backward_fn(grad):
            return [np.ones_like(t.data) * grad]
        out.backward_fn = backward_fn
    return out

def mean(t, axis = 0, keepdims=True):
    n = t.data.size
    out = Tensor(np.sum(t.data, axis=axis, keepdims=keepdims)/n, requires_grad=t.requires_grad)
    if t.requires_grad:
        out.parents = [t]
        def backward_fn(grad):            
            return [np.ones_like(t.data) * grad/n]
        out.backward_fn = backward_fn
    return out

def matmul(t1, t2):
    requires_grad = t1.requires_grad or t2.requires_grad
    out = Tensor(np.matmul(t1.data, t2.data), requires_grad=requires_grad)
    if requires_grad:
        out.parents = [t1, t2]
        def backward_fn(grad):            
            return [np.matmul(grad, t2.data.T), np.matmul(t1.data.T, grad)]
        out.backward_fn = backward_fn
    return out

def sqrt(t):
    out = Tensor(np.sqrt(t.data), requires_grad=t.requires_grad)
    if t.requires_grad:
        out.parents = [t]
        def backward_fn(grad):            
            return [grad / (2 * np.sqrt(t.data))]
        out.backward_fn = backward_fn
    return out


# b.data = b.data - (lr * b.grad)
# b.grad = None
