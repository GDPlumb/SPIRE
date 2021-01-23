
class Features:
    def __init__(self, requires_grad = None):
        self.features = None
        self.requires_grad = requires_grad
        
    def __call__(self, modules, module_in, module_out):
        if self.requires_grad is not None:
            module_out.requires_grad = self.requires_grad
        self.features = module_out
