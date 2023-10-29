from system.pendulum import InvertedPendulumSystem, InvertedPendulumSystemCART
import numpy as np
from typing import Tuple

class Simulator:

    def __init__(
        self,
        system: InvertedPendulumSystem,
        N_steps: int,
        step_size: float,
        state_init: np.array,
    ):
        self.system = system
        self.N_steps = N_steps
        self.step_size = step_size
        self.state = np.copy(state_init)
        self.state_init = np.copy(state_init)
        self.current_step_idx = 0

    def step(self) -> bool:

        if self.current_step_idx <= self.N_steps:

            self.state += (
                self.system.compute_closed_loop_rhs(self.state) * self.step_size
            )
            
            self.current_step_idx += 1
            return True
        else:
            return False

    def reset(self) -> None:

        self.state = np.copy(self.state_init)
        self.current_step_idx = 0
        self.system.reset()

    def get_sim_step_data(self) -> Tuple[np.array, np.array, int]:

        return (
            self.system.get_observation(self.state),
            np.copy(self.system.action),
            int(self.current_step_idx),
        )
    
class SimulatorLimited(Simulator):
    def __init__(self, system: InvertedPendulumSystem, 
                 N_steps: int, 
                 step_size: float, 
                 state_init: np.array,
                 max_distance: float = 10):
        super().__init__(system, N_steps, step_size, state_init)
        self.MAX_DISTANCE = max_distance

    def step(self) -> bool:
        if self.current_step_idx <= self.N_steps:

            self.state += (
                self.system.compute_closed_loop_rhs(self.state) * self.step_size
            )
            if self.state[2] > self.MAX_DISTANCE:
                self.state[2] = self.MAX_DISTANCE
                self.state[3] = 0
            elif self.state[2] < -self.MAX_DISTANCE:
                self.state[2] = -self.MAX_DISTANCE
                self.state[3] = 0
            
            self.current_step_idx += 1
            return True
        else:
            return False
