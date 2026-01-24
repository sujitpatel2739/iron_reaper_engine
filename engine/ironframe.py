import numpy as np

class Tensor:
    def __init__(self, data, requires_grad = False):
        self.grad = None
        self.data = np.array(data)
        self.parents = []
        self.requires_grad = requires_grad
        self.backward_fn = None
    
    def backward(self, grad=None):
        if not self.requires_grad:
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
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return mul(self, other)
    
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


# DRIVER CODE ----------------------------------------------------------------

# m samples = 1000
# m = 1000
# n Features = 100
# n = 100 
# W = Tensor(np.random.randn(n, 1) * 0.01, requires_grad=True)
# b = Tensor(np.zeros((1, 1)), requires_grad=True)

# def linear_forward(X_np, W, b):
#     X = Tensor(X_np, requires_grad=False)
#     return add(matmul(X, W), b)

def mse_loss(y_pred, y_true_np):
    y_true = Tensor(y_true_np, requires_grad=False)

    neg_y = mul(y_true, Tensor(-1.0))
    diff = add(y_pred, neg_y)
    diff_sq = mul(diff, diff)

    return mean(diff_sq)

# # forward   
# X_np = np.array(np.random.randn(m, n)*0.01)
# y_true_np = np.array(np.random.randn(m, 1)*0.01)
# y_pred = linear_forward(X_np, W, b)
# loss = mse_loss(y_pred, y_true_np)

# # backward
# loss.backward()

# # update (SGD, lr = 0.01)
# lr = 0.01

# W.data = W.data - (lr * W.grad)
# W.grad = None

# b.data = b.data - (lr * b.grad)
# b.grad = None