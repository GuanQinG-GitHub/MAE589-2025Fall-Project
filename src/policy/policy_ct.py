# -*- coding: utf-8 -*-
"""LSPI Policy class for continuous state/action spaces."""
import random
import numpy as np
from scipy.optimize import minimize
import os
import sys
# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from policy import Policy
from basis_functions import QuadraticBasisFunction
from qpsolvers import solve_qp


class QuadraticPolicy(Policy):
    """Implements LSPI policy with quadratic programming for continuous control."""
    
    def __init__(self, n_state, n_action, discount=1.0, explore=0.0, weights=None, 
                  folder_path = None):
        self.folder_path = folder_path
        self.n_action = n_action
        self.n_state = n_state
        self.n_basis = int((n_action + n_state)*(n_state + n_action + 1)/2)
        # print(f"Number of basis functions: {self.n_basis}")
        self.basis = QuadraticBasisFunction(n_state, n_action)
 

        if folder_path is not None:
            # load scaling and offset for policy 
            self.offset_a = 0 #np.load(os.path.join(folder_path, "offset_a.npy"))
            self.scale_a = np.load(os.path.join(folder_path,"scale_a.npy"))
            self.offset_s = np.load(os.path.join(folder_path,"offset_s.npy"))
            self.scale_s = np.load(os.path.join(folder_path,"scale_s.npy"))
        else:
            self.scale_s = 1.
            self.scale_a = 1.
            self.offset_s = 0.
            self.offset_a = 0.
        if weights is None:
            w = np.random.uniform(0., 1.0, size=(self.n_basis,))
            # Ensure weights are positive definite in state-action space
            S = convertW2S(w)  # convert weights to state-action space
            D,V = np.linalg.eig(S)
            D[D < 0] = 0  # ensure positive definiteness
            S = V.T @ np.diag(D) @ V  # reconstruct positive definite matrix
            weights = convertS2W(S)  # convert back to weight vector
        
        super().__init__(self.basis, discount, explore, weights)
        

        
    def cp(self):
        """Return a copy of this class with a deep copy of the weights."""
        return QuadraticPolicy(self.n_state, 
                        self.n_action,
                        self.discount,
                        self.explore,
                        np.copy(self.weights),
                        self.folder_path,
                       )
    
    def calc_q_value(self, state, action):
        # scale and offset state and action
        state = (state - self.offset_s) / self.scale_s
        action = (action - self.offset_a) / self.scale_a
        """Calculate Q-value for a given state-action pair."""
        return self._calc_q_value(state, action)
    
    def _calc_q_value(self, state, action): # q value should be all positive 
        # if action.shape[0] < 0 or action >= self.n_action:
        #     raise IndexError('action must be in range [0, num_actions)')
    
        # self.Huu, self.Hf = self.extract_qp_parameters(state)
        # q = 0 
        # # q += 0.5 * action[None,:] @ self.Huu @ action[:,None]  # for testing
        # q += self.Hf @ action[:,None] # linear term
        # return q 
        return self.basis.evaluate(state, action) @ self.weights

    def select_action(self, state):
        """Select action with ε-greedy exploration."""
        if random.random() < self.explore:
            return np.random.uniform(low=-0.5, high=0.5, size=self.n_action) * self.scale_a + self.offset_a
        return self.best_action(state)
    
    def best_action(self, state): # best action to reduce Q value
        """apply scaling and offsetting of action and state"""
        state = (state - self.offset_s) / self.scale_s
        action = self._best_action(state)
        action = action * self.scale_a + self.offset_a
        return action
    

    def _best_action_old(self, state):
        # self.Huu, self.Hf = self.extract_qp_parameters(state) 
        q_func = lambda a: self.basis.evaluate(state, a) @ self.weights
    
        res = minimize(
            fun=q_func, 
            # fun = lambda a: self.Hf @ a[:,None] ,
            x0=np.zeros((self.n_action,)),
            bounds=[(-1,1.)], # for a N(0,1) distribution, it is reasonable to set the bounds to (-3, 3), because 99.7% of the values will fall within this range
            # constraints={'type': 'ineq', 'fun': lambda a: A @ a - Ax}
        )
        action = res.x

        return action

    def _best_action(self, state):
        """Solve quadratic program for optimal action using qpsolvers"""
        Huu, Hf = self.extract_qp_parameters(state)
        
        # QP formulation: 0.5*a^T*P*a + q^T*a
        P = Huu.astype(np.float64)
        q = Hf.astype(np.float64)
        
        # Action constraints: -1 <= a <= 1
        n_action = self.n_action
        G = np.vstack([np.eye(n_action), -np.eye(n_action)])  # a <= 1 and -a <= 1
        h = np.hstack([np.ones(n_action), np.ones(n_action)])*0.2 # Upper/lower bounds
        
        # Solve QP with tolerance for numerical stability
        action = solve_qp(
            P=P,
            q=q,
            G=G,
            h=h,
            solver='osqp',  # Best for convex problems
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=100000
        )
        
        if action is None:  # Fallback if QP fails
            return np.zeros(self.n_action)
            
        return action

    def extract_qp_parameters(self, state):
        """Solve quadratic program for optimal action with constraints."""
        nS = self.n_state  # State dimensions
        nSA = self.n_state + self.n_action  # State+action dimensions
        
        HW = convertW2S(self.weights)
        # diagonal element should only be half of the current value


        # Extract weight submatrices
        Hux = HW[nS:nSA, :nS] 
        Huu = HW[nS:nSA, nS:nSA]
        Hxu = HW[:nS, nS:nSA]
        
        # Quadratic programming setup
        Hf = state.T @ Hxu
        return Huu,Hf

    def _apply_action_constraints(self, action):
        """Enforce physical constraints on actions."""
        y1, y2 = self.first_stiff_timing, self.second_stiff_timing
        
        if action[0] + y1 < 0.01:
            action[0] = 2 * abs(y1 - 0.01)
            
        if (y2 + action[1] - y1 - action[0]) < 0:
            action[1] = abs(y2 - y1 - action[0]) * 2
            
        if action[1] + y2 > 0.98:
            action[1] = -2 * abs(0.98 - y2)
            
        return np.clip(action, -1, 1)

    def evaluate_basis(self, state, action):
        """Evaluate basis function for given state-action pair."""
        return self.basis(state, action)


    def cvxw_update(self, experience, start_idx, end_idx, tol=1e-4, verbose = True):
        """
        Update policy weights using convex optimization with experience replay
        
        Args:
            experience: Dict with keys:
                'states': np.array of shape (n_state, buffer_size)
                'actions': np.array of shape (n_action, buffer_size)
                'costs': np.array of shape (buffer_size,)
                'next_states': np.array of shape (n_state, buffer_size)
            start_idx: Starting index in experience buffer
            end_idx: Ending index in experience buffer
            tol: Convergence tolerance
        """
        # Extract experience window
        for k, v in experience.items():
            arr = np.array(v)
            experience[k] = arr
            # print(f"Experience {k} shape: {arr.shape}, start_idx: {start_idx}, end_idx: {end_idx}")
            
        states = experience['current_state'][ start_idx:end_idx]
        actions = experience['action'][ start_idx:end_idx]
        costs = experience['stage_cost'][start_idx:end_idx]
        next_states = experience['next_state'][ start_idx:end_idx]
        
        if verbose:
            print("Experience replay window stats:")
            print("start_idx", start_idx, "end_idx", end_idx)
            print("stage cost", costs)
            print("qvalues", [self.calc_q_value(s, a) for s, a in zip(states, actions)])
            # print("states", states)
        # Convert to policy's scaled representation
        states = (states - self.offset_s) / self.scale_s
        actions = (actions - self.offset_a) / self.scale_a
        
        # Initialize arrays
        num_samples = end_idx - start_idx
        pi_actions = np.zeros((self.n_action, num_samples))
        importance_ratios = np.zeros(num_samples)
        q_states = np.zeros((self.n_basis, num_samples))
        q_next_states = np.zeros((self.n_basis, num_samples))

        # Action optimization loop
        for i in range(num_samples):
            # Get current QP parameters
            # Huu, Hf = self.extract_qp_parameters(states[i])
            
            # # Solve QP using qpsolvers
            # P = Huu.astype(np.double)
            # q = Hf.astype(np.double)
            # G = np.vstack([np.eye(self.n_action), -np.eye(self.n_action)]) 
            # h = np.hstack([np.ones(self.n_action), np.ones(self.n_action)]) *0.3  # Action bounds
            # pi_actions[:,i] = solve_qp(P, q, G,h, solver='osqp')
            pi_actions[:,i] = self._best_action(states[i])  # Get best action for current state
            # Construct basis features
            q_states[:,i] = self.basis.evaluate(states[i], actions[i])
            q_next_states[:,i] = self.basis.evaluate(next_states[i], pi_actions[:,i])
            
            # Importance sampling ratio (adapt based on your logic)
            importance_ratios[i] = 1.0 # if np.random.rand() < 0.95 else 2.0

        # Matrix construction for least squares
        W = np.diag(importance_ratios)
        Cphi = (q_states @ q_states.T - self.discount * q_states @ W @ q_next_states.T) / num_samples
        dphi = (q_states @ W @ costs) / num_samples

        # Weight update with Dykstra projection
        phiw = self.weights.copy()
        error = np.inf
        step_size = 0.5
        max_iter = 1000
        
        for iter in range(max_iter):
            if error < tol:
                break
            
            phiw_prev = phiw.copy()
            residual = phiw - step_size * (Cphi @ phiw - dphi)
            phiw = proDykstra(residual, 1e7, errTol=tol)
            
            error = np.linalg.norm(phiw_prev - phiw)
            step_size = 1 / (iter + 2)

        # Update policy weights while maintaining symmetry
        S = 0.5 * (convertW2S(phiw) + convertW2S(phiw).T)
        self.weights = convertS2W(S)  # Convert back to weight vector

 
# alternating projection
def proDykstra(x0,ballR,errTol):
    """Projection onto the set of symmetric matrices."""    

    error=1
    j=1
    I= np.zeros((len(x0),2))
    oldI=np.zeros((len(x0),2))
    x=x0.reshape((-1,))
    while j<500 and error>errTol: 
        
        oldX=x.copy()
        if np.linalg.norm(x-I[:,0])>ballR:
            x=ballR*(x-I[:,0])/np.linalg.norm(x-I[:,0])                    
        else:
            x=x-I[:,0]               
        oldI[:,0]=I[:,0].copy()
        I[:,0]=x-(oldX-I[:,0])
        
        oldX=x.copy()
        s=convertWS(x-I[:,1])
        D, V = np.linalg.eig(s)  # D is diagonal matrix, V is orthogonal
        D[D< 0]=0    # set negative eigenvalues to zero                          
        s=V@np.diag(D)@V.T
        x=convertSW(s) # x is the new point
        oldI[:,1]=I[:,1].copy()
        I[:,1]=x-(oldX-I[:,1])  
                        
        j=j+1
        error=np.linalg.norm(oldI-I)**2                
            
    return x # return the projection of x0 onto the set of symmetric matrices
   
def convertWS(w):
    """
    Convert a weight vector w to a symmetric matrix S.
    The length of w should be n*(n+1)/2 for some integer n.
    """
    n = int((np.sqrt(1 + 8 * len(w)) - 1) / 2)
    S = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            S[i, j] = w[idx]
            S[j, i] = w[idx]
            idx += 1
    # Adjust diagonal elements (if needed, depending on your convention)
    # S = S - 0.5 * np.diag(np.diag(S))
    return S

def convertSW(S):
    """
    Convert a symmetric matrix S to a vector W (upper triangular, row-wise).
    """
    n = S.shape[0]
    W = []
    for i in range(n):
        for j in range(i, n):
            W.append(S[i, j])
    return np.array(W)

def convertW2S(w):
    """Convert weight vector to the corresponding LQR matrix."""
    # Size of the symmetric matrix
    n = int((np.sqrt(1 + 8 * len(w)) - 1) / 2) 
    idx = 0
    Phat = np.zeros((n, n))
    
    # Fill the upper triangular part of the matrix
    for r in range(n):
        for c in range(r, n):
            Phat[r, c] = w[idx]
            idx += 1
    
    # Symmetrize the matrix
    S = 0.5 * (Phat + Phat.T)
    return S



def convertS2W(S):
    """Convert LQR matrix to weight vector."""
    if not np.allclose(S, S.T):
        raise ValueError("Input error: S must be a symmetric matrix.")
    
    n = S.shape[0]
    W = []
    for i in range(n):
        for j in range(i, n):
            if i == j:
                W.append(S[i, j])
            else:
                W.append(S[i, j]*2)
    return np.array(W)


if __name__ == "__main__":
    # Example usage
    n_state = 3
    n_action = 2
    policy = QuadraticPolicy(n_state, n_action)
    
    # Example state and action
    state = np.array([0.5, 0.2, 0.1])
    action = np.array([0.1, -0.1])
    
    q_value = policy.calc_q_value(state, action)
    print("Q-value:", q_value)
    
    best_action = policy.best_action(state)
    print("Best action:", best_action)
    

    assert np.allclose(policy.basis.evaluate(state, action) @ policy.weights, policy._calc_q_value(state, action)), "Basis evaluation mismatch"