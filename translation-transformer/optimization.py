class ScheduledOptim:
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=2000):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.current_lr = init_lr
    
    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(self.n_steps ** -0.5, self.n_steps * (self.n_warmup_steps ** -1.5))

    def update_learning_rate(self):
        self.n_steps += 1
        self.current_lr = self.init_lr * self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

    @property
    def get_current_lr(self):
        return self.current_lr