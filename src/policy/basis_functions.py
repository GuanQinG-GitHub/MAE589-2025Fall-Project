import numpy as np

class QuadraticBasisFunction:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action

    def evaluate(self, state, action):
        """
        Compute quadratic basis features for state-action pair.
        Phi(s, a) should correspond to the weights structure in policy_ct.py.
        The policy uses a symmetric matrix S constructed from weights.
        Q(s, a) = [s, a]^T S [s, a]
        So the basis functions are the unique terms in the outer product of [s, a].
        """
        # Concatenate state and action
        sa = np.concatenate([state, action])
        n = len(sa)
        
        # Compute unique quadratic terms
        features = []
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    features.append(sa[i] * sa[j])
                else:
                    features.append(2 * sa[i] * sa[j]) # Factor of 2 for off-diagonal to match S structure
        
        return np.array(features)
