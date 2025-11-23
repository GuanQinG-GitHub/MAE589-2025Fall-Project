class Policy:
    def __init__(self, basis, discount, explore, weights):
        self.basis = basis
        self.discount = discount
        self.explore = explore
        self.weights = weights

    def select_action(self, state):
        raise NotImplementedError

    def best_action(self, state):
        raise NotImplementedError
