from scipy.linalg import solve_continuous_are
import numpy as np

class LQR:
    def __init__(self, m, mc, l, g):
        A = np.array([[0, 1],[(m+mc)*g/mc*l, 0]])
        B = np.array([[0], [-1/(l*mc)]])
        Q = np.eye(2)
        R = np.array([1])
        P = solve_continuous_are(A, B, Q, R)
        self.K = B.T @ P
        
        
    
    def state_dynamics_function(self, x, u):
        return self.A @ x + self.B @ u
    
    def get_action(self, x):
        return -10*self.K@x

