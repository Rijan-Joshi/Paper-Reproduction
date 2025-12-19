from src.utils.cupy_numpy import np

class SDLM:
    """
        Stochastic Diagonal Levenbeg Marquardt Method Implementation for Optimization of the Network
    """

    def __init__(self, lr = 1.0, mu = 0.01, decay = 0.05, eps = 1e-8):
        self.lr = lr
        self.damping = mu
        self.weight_decay = decay
        self.eps = eps
        self.hessian_diag = {}
        self.lr_multipliers = {}
    
    def set_learning_rate(self, name, lr):
        self.lr_multipliers[name] = lr
    
    def step(self, named_params, grads):
        for (name, param), grad in zip(named_params, grads):
            if name not in self.hessian_diag:
                self.hessian_diag[name] = np.zeros_like(param)
            
            self.hessian_diag[name] = (
                self.weight_decay * self.hessian_diag[name] + (1 - self.weight_decay) * (grad ** 2)
            )

            lr = self.lr_multipliers.get(name, self.lr)

            #SDLM Update
            denom  = self.damping + np.sqrt(self.hessian_diag[name]) + self.eps
            param -= lr * grad/ denom

