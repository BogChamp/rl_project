import numpy as np

class InvertedPendulumSystem:
    dim_action: int = 1
    dim_observation: int = 3
    dim_state: int = 2

    m: float = 0.1
    mc: float = 1
    l: float = 0.5
    g: float = 9.81

    def __init__(self, actor_type, f:float = 10.0) -> None:
        self.f = f
        assert actor_type in ["discrete", "continuous"]
        self.actor_type = actor_type
        self.reset()

    def reset(self) -> None:

        self.action = np.zeros(self.dim_action)
    
    def get_action(self, action):
        if self.actor_type == "discrete":
            return (action - 1) * self.f
        elif self.actor_type == "continuous":
            return action

    def compute_dynamics(self, state: np.array, action: np.array) -> np.array:

        Dstate = np.zeros(self.dim_state)

        sin_angle = np.sin(state[0])
        cos_angle = np.cos(state[0])
        Dstate[0] = state[1]
        u = self.get_action(action)
        Dstate[1] = (self.m + self.mc) * self.g * sin_angle - cos_angle * (u + self.m * self.l * state[1]**2 * sin_angle)
        Dstate[1] /= 4 / 3 * (self.m + self.mc) * self.l - self.m * self.l * cos_angle**2

        return Dstate

    def compute_closed_loop_rhs(self, state: np.array) -> np.array:

        system_right_hand_side = self.compute_dynamics(state, self.action)
        return system_right_hand_side

    def receive_action(self, action: np.array) -> None:

        self.action = action

    @staticmethod
    def get_observation(state: np.array) -> np.array:
        observation = np.zeros(InvertedPendulumSystem.dim_observation)
        observation[0] = np.cos(state[0])
        observation[1] = np.sin(state[0])
        observation[2] = state[1]
        return observation

class InvertedPendulumSystemCART(InvertedPendulumSystem):
    dim_observation: int = 5
    dim_state: int = 4

    def compute_dynamics(self, state: np.array, action: np.array) -> np.array:

        Dstate = np.zeros(self.dim_state)

        sin_angle = np.sin(state[0])
        cos_angle = np.cos(state[0])
        Dstate[0] = state[1]
        u = self.get_action(action)
        Dstate[1] = (self.m + self.mc) * self.g * sin_angle - cos_angle * (u + self.m * self.l * state[1]**2 * sin_angle)
        Dstate[1] /= 4 / 3 * (self.m + self.mc) * self.l - self.m * self.l * cos_angle**2
        Dstate[2] = state[3]
        Dstate[3] = (action + self.m * self.l * (state[1]**2 * sin_angle - Dstate[1]*cos_angle)) / (self.m + self.mc)

        return Dstate

    @staticmethod
    def get_observation(state: np.array) -> np.array:
        observation = np.zeros(InvertedPendulumSystemCART.dim_observation)
        observation[0] = np.cos(state[0])
        observation[1] = np.sin(state[0])
        observation[2] = state[1]
        observation[3] = state[2]
        observation[4] = state[3]

        return observation

class InvertedPendulumSystemLQR(InvertedPendulumSystem):
    dim_observation: int = 2
    dim_state: int = 2

    @staticmethod
    def get_observation(state: np.array) -> np.array:
        observation = state
        return observation
