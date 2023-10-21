from torch import nn
import torch
from optimizer import Optimizer
from buffer import IterationBuffer
from typing import Dict

class PolicyREINFORCE:
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, device: str = "cpu", is_with_baseline: bool = True,
    ) -> None:
        """Initialize policy

        Args:
            model (nn.Module): model to optimize
            optimizer (Optimizer): optimizer for `model` weights optimization
            device (str, optional): device for gradient descent optimization procedure. Defaults to "cpu".
            is_with_baseline (bool, optional): whether to use baseline in objective function.
        """

        self.buffer = IterationBuffer()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.is_with_baseline = is_with_baseline

    def objective(self, batch: Dict["str", torch.tensor]) -> torch.tensor:
        """This method computes a proxy objective specifically for automatic differentiation since its gradient is exactly as in REINFORCE

        Args:
            batch (torch.tensor): batch with catted observations-actions, total objectives and baselines

        Returns:
            torch.tensor: objective value
        """

        observations_actions = batch["observations_actions"].to(self.device)
        tail_total_objectives = batch["tail_total_objectives"].to(self.device)
        baselines = batch["baselines"].to(self.device)
        N_episodes = self.N_episodes


        log_probs = self.model.log_probs(observations_actions.float())
        if self.is_with_baseline:
            tail_total_objectives -= baselines
        
        tail_total_objectives *= log_probs
        #print('reinforce_obj', tail_total_objectives.sum() / N_episodes)
        return tail_total_objectives.sum() / N_episodes


    def REINFORCE_step(self) -> None:
        """Do gradient REINFORCE step"""

        self.N_episodes = self.buffer.get_N_episodes()
        self.model.to(self.device)
        self.optimizer.optimize(self.objective, self.buffer)
        self.model.to("cpu")
        self.buffer.nullify_buffer()