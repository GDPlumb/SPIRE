
class Features:
    def __init__(self, requires_grad = False):
        self.features = None
        self.requires_grad = requires_grad
        
    def __call__(self, modules, module_in, module_out):
        module_out.requires_grad = self.requires_grad
        self.features = module_out
