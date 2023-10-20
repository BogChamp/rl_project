import numpy as np

class InvertedPendulumSystem:
    dim_action: int = 1
    dim_observation: int = 4
    dim_state: int = 4

    m: float = 0.1
    mc: float = 1
    l: float = 0.5
    g: float = 9.81

    def __init__(self) -> None:

        self.reset()

    def reset(self) -> None:

        self.action = np.zeros(self.dim_action)

    def compute_dynamics(self, state: np.array, action: np.array) -> np.array:

        Dstate = np.zeros(self.dim_observation)

        sin_angle = np.sin(state[0])
        cos_angle = np.cos(state[0])
        Dstate[0] = state[1]
        Dstate[1] = (self.m + self.mc) * self.g * sin_angle - cos_angle * (action + self.m * self.l * state[1]**2 * sin_angle)
        Dstate[1] /= 4 / 3 * (self.m + self.mc) * self.l - self.m * self.l * cos_angle**2
        Dstate[2] = state[3]
        Dstate[3] = (action + self.m * self.l * (state[1]**2 * sin_angle - Dstate[1]*cos_angle)) / (self.m + self.mc)

        return Dstate

    def compute_closed_loop_rhs(self, state: np.array) -> np.array:

        system_right_hand_side = self.compute_dynamics(state, self.action)
        return system_right_hand_side

    def receive_action(self, action: np.array) -> None:

        self.action = action

    @staticmethod
    def get_observation(state: np.array) -> np.array:
        observation = state

        return observation