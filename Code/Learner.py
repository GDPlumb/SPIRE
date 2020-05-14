
class Region:
    def __init__(self, base_model, encoder):
        self.base_model = base_model()
        self.encoder = encoder
        
    def fit(self, X, y):
        if self.encoder is None:
            self.base_model.fit(X, y)
        else:
            self.base_model.fit(self.encoder(X).numpy(), y)
            
    def predict(self, X):
        if self.encoder is None:
            return self.base_model.predict(X)
        else:
            return self.base_model.predict(self.encoder(X).numpy())
            
class Learner:
    def __init__(self, base_model, encoder = None):
        self.base_model = base_model
        self.encoder = encoder
        
    def new(self):
        return Region(self.base_model, self.encoder)
