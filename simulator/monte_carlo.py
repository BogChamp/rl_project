import numpy as np
import torch
from typing import Callable
from IPython.display import clear_output
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from system.simulator import Simulator
from system.pendulum import InvertedPendulumSystem
from policy.policy_reinforce import PolicyREINFORCE
from models.LQR import LQR


class MonteCarloSimulationScenario:
    """Run whole REINFORCE procedure"""

    def __init__(
        self,
        simulator: Simulator,
        system: InvertedPendulumSystem,
        policy: PolicyREINFORCE,
        N_episodes: int,
        N_iterations: int,
        discount_factor: float = 1.0,
        termination_criterion: Callable[
            [np.array, np.array, float, float], bool
        ] = lambda *args: False,
    ):
        """Initialize scenario for main loop


        Args:
            simulator (Simulator): simulator for computing system dynamics
            system (InvertedPendulumSystem): system itself
            policy (PolicyREINFORCE): REINFORCE gradient stepper
            N_episodes (int): number of episodes in one iteration
            N_iterations (int): number of iterations
            discount_factor (float, optional): discount factor for running objectives. Defaults to 1
            termination_criterion (Callable[[np.array, np.array, float, float], bool], optional): criterion for episode termination. Takes observation, action, running_objective, total_objectove. Defaults to lambda*args:False
        """

        self.simulator = simulator
        self.system = system
        self.policy = policy
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.termination_criterion = termination_criterion
        self.discount_factor = discount_factor

        self.total_objective = 0
        self.total_objectives_episodic = []
        self.learning_curve = []
        self.last_observations = None

    def compute_running_objective(
        self, observation: np.array, action: np.array
    ) -> float:
        """Computes running objective

        Args:
            observation (np.array): current observation
            action (np.array): current action

        Returns:
            float: running objective value
        """
        # out = 50*(1-observation[0])
        # if observation[0] > 0.995:
        #     out += observation[2]**2 + 10*observation[1]**2
        # return out
        # out = observation[3]**2
        # if observation[0] < -0.5:
        #     return 1000 + observation[3]**2
        # elif observation[0] < 0.8:
        #     return 500 + observation[3]**2
        # elif observation[0] < 0.95:
        #     return 200
        # else:
        #     return observation[2]**2
        #print(observation[0])
        # if np.abs(observation[3]) > 5:
        #     return 100*(1-observation[0]) + observation[3]**2
        # return 100*(1-observation[0])
        # if observation[0] < 0.9:
        #     return 80*(1-observation[0]) + observation[1]**2 + observation[3]**2
        # else:
        return 30*(1-observation[0]) + observation[1]**2
    
    def run(self) -> None:
        """Run main loop"""

        eps = 0.1
        means_total_objectives = [eps]
        for iteration_idx in range(self.N_iterations):
            if iteration_idx % 10 == 0:
                clear_output(wait=True)
            for episode_idx in tqdm(range(self.N_episodes)):
                terminated = False
                while self.simulator.step():
                    (
                        observation,
                        action,
                        step_idx,
                    ) = self.simulator.get_sim_step_data()

                    new_action = (
                        self.policy.model.sample(torch.tensor(observation).float())
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    discounted_running_objective = self.discount_factor ** (
                        step_idx
                    ) * self.compute_running_objective(observation, new_action)
                    self.total_objective += discounted_running_objective

                    if not terminated and self.termination_criterion(
                        observation,
                        new_action,
                        discounted_running_objective,
                        self.total_objective,
                    ):
                        terminated = True

                    if not terminated:
                        self.policy.buffer.add_step_data(
                            np.copy(observation),
                            np.copy(new_action),
                            np.copy(discounted_running_objective),
                            step_idx,
                            episode_idx,
                        )
                    self.system.receive_action(new_action)
                #print("A", len(self.policy.buffer.observations), len(self.policy.buffer.actions), len(self.policy.buffer.running_objectives))
                self.simulator.reset()
                self.total_objectives_episodic.append(self.total_objective)
                self.total_objective = 0
            self.learning_curve.append(np.mean(self.total_objectives_episodic))
            self.last_observations = pd.DataFrame(
                index=self.policy.buffer.episode_ids,
                data=self.policy.buffer.observations.copy(),
            )
            self.last_actions = pd.DataFrame(
                index=self.policy.buffer.episode_ids,
                data=self.policy.buffer.actions.copy(),
            )
            self.policy.REINFORCE_step()

            means_total_objectives.append(np.mean(self.total_objectives_episodic))
            change = (means_total_objectives[-1] / means_total_objectives[-2] - 1) * 100
            sign = "-" if np.sign(change) == -1 else "+"
            print(
                f"Iteration: {iteration_idx + 1} / {self.N_iterations}, "
                + f"mean total cost {round(means_total_objectives[-1], 2)}, "
                + f"% change: {sign}{abs(round(change,2))}, "
                + f"last observation: {self.last_observations.iloc[-1].values.reshape(-1)}",
                end="\n",
            )

            self.total_objectives_episodic = []

    def plot_data(self):
        """Plot learning results"""

        data = pd.Series(
            index=range(1, len(self.learning_curve) + 1), data=self.learning_curve
        )
        na_mask = data.isna()
        not_na_mask = ~na_mask
        interpolated_values = data.interpolate()
        interpolated_values[not_na_mask] = None
        data.plot(marker="o", markersize=3)
        interpolated_values.plot(linestyle="--")

        plt.title("Total cost by iteration")
        plt.xlabel("Iteration number")
        plt.ylabel("Total cost")
        #plt.yscale("log")
        plt.show()


        cos_theta_ax, sin_theta_ax, dot_theta_ax = pd.DataFrame(
            data=self.last_observations.loc[0].values
        ).plot(
            xlabel="Step Number",
            title="Observations in last iteration",
            legend=False,
            subplots=True,
            grid=True,
        )
        cos_theta_ax.set_ylabel("cos angle")
        sin_theta_ax.set_ylabel("sin angle")
        dot_theta_ax.set_ylabel("angular velocity")
        # h_ax.set_ylabel("cartpole coord")
        # dot_h_ax.set_ylabel("cartpole velocity")

        actions_ax = pd.DataFrame(
            data=self.last_actions.loc[0].values
        ).plot(
            xlabel="Step Number",
            title="Actions in last iteration",
            legend=False,
            grid=True,
        )
        actions_ax.set_ylabel("action")

        plt.show()


class MonteCarloLQR(MonteCarloSimulationScenario):
    def __init__(
        self,
        simulator: Simulator,
        system: InvertedPendulumSystem,
        lqr: LQR, 
        N_episodes: int,
        N_iterations: int,
        discount_factor: float = 1.0,
        termination_criterion: Callable[
            [np.array, np.array, float, float], bool
        ] = lambda *args: False,
    ):
        self.simulator = simulator
        self.system = system
        self.lqr = lqr
        self.N_episodes = N_episodes
        self.N_iterations = N_iterations
        self.termination_criterion = termination_criterion
        self.discount_factor = discount_factor

        self.total_objective = 0
        self.total_objectives_episodic = []
        self.learning_curve = []
        self.last_observations = None
    
    def run(self) -> None:
        """Run main loop"""

        eps = 0.1
        means_total_objectives = [eps]
        for iteration_idx in range(self.N_iterations):
            if iteration_idx % 10 == 0:
                clear_output(wait=True)
            observations = []
            actions = []
            for episode_idx in tqdm(range(self.N_episodes)):
                terminated = False
                while self.simulator.step():
                    (
                        observation,
                        action,
                        step_idx,
                    ) = self.simulator.get_sim_step_data()

                    new_action = self.lqr.get_action(observation)
                    discounted_running_objective = self.discount_factor ** (
                        step_idx
                    ) * self.compute_running_objective(observation, new_action)
                    self.total_objective += discounted_running_objective

                    if not terminated and self.termination_criterion(
                        observation,
                        new_action,
                        discounted_running_objective,
                        self.total_objective,
                    ):
                        terminated = True
                    
                    observations.append(np.copy(observation))
                    actions.append(new_action)
                    self.system.receive_action(new_action)
                #print("A", len(self.policy.buffer.observations), len(self.policy.buffer.actions), len(self.policy.buffer.running_objectives))
                self.simulator.reset()
                self.total_objectives_episodic.append(self.total_objective)
                self.total_objective = 0
            self.learning_curve.append(np.mean(self.total_objectives_episodic))
            self.last_observations = pd.DataFrame(observations)
            self.last_actions = pd.DataFrame(actions)

            means_total_objectives.append(np.mean(self.total_objectives_episodic))
            change = (means_total_objectives[-1] / means_total_objectives[-2] - 1) * 100
            sign = "-" if np.sign(change) == -1 else "+"
            print(
                f"Iteration: {iteration_idx + 1} / {self.N_iterations}, "
                + f"mean total cost {round(means_total_objectives[-1], 2)}, "
                + f"% change: {sign}{abs(round(change,2))}, "
                + f"last observation: {self.last_observations.iloc[-1].values.reshape(-1)}",
                end="\n",
            )

            self.total_objectives_episodic = []
        
    def plot_data(self):
        """Plot learning results"""

        self.last_observations.plot(subplots=True)

        plt.show()