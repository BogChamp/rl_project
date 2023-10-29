import numpy as np
from torch import nn
import torch
from typing import Dict, Optional, Callable, Type, Any
from torch.utils.data import DataLoader
from buffer.buffer import IterationBuffer

class LRSchedulerSwitch:
    """Callable class that returns True in case ||observation|| <= norm_observation_threshold"""

    def __init__(self, norm_observation_threshold: float) -> None:
        """Initialize LRSchedulerSwitch.

        Args:
            norm_observation_threshold (float): threshold for observation norm
        """
        self.norm_observation_threshold = norm_observation_threshold
        self.turned_on = False

    def __call__(self, observation: np.array) -> bool:
        """Return True if ||observation|| <= norm_observation_threshold

        Args:
            observation (np.array): observation

        Returns:
            bool: ||observation|| <= norm_observation_threshold
        """

        if (
            self.turned_on
            or np.linalg.norm(observation) <= self.norm_observation_threshold
        ):
            self.turned_on = True
            return True
        else:
            return False


class Optimizer:
    """Does gradient step for optimizing model weights"""

    def __init__(
        self,
        model: nn.Module,
        opt_method: Type[torch.optim.Optimizer],
        opt_options: Dict[str, Any],
        lr_scheduler_method: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        lr_scheduler_options: Optional[Dict[str, Any]] = None,
        lr_scheduler_switch: Callable[[np.array], bool] = lambda _: True,
        shuffle: bool = True,
    ):
        """Initialize Optimizer

        Args:
            model (nn.Module): model which weights we need to optimize
            opt_method (Type[torch.optim.Optimizer]): method type for optimization. For instance, `opt_method=torch.optim.SGD`
            opt_options (Dict[str, Any]): kwargs dict for opt method
            lr_scheduler_method (Optional[torch.optim.lr_scheduler.LRScheduler], optional): method type for LRScheduler. Defaults to None
            lr_scheduler_options (Optional[Dict[str, Any]], optional): kwargs for LRScheduler. Defaults to None
            lr_scheduler_switch (Callable[[np.array], bool]): callable object for turning on the sheduller. Defaults to lambda _: True
            shuffle (bool, optional): whether to shuffle items in dataset. Defaults to True
        """

        self.opt_method = opt_method
        self.opt_options = opt_options
        self.shuffle = shuffle
        self.model = model
        self.optimizer = self.opt_method(self.model.parameters(), **self.opt_options)
        self.lr_scheduler_method = lr_scheduler_method
        self.lr_scheduler_options = lr_scheduler_options
        self.lr_scheduler_switch = lr_scheduler_switch
        if self.lr_scheduler_method is not None:
            self.lr_scheduler = self.lr_scheduler_method(
                self.optimizer, **self.lr_scheduler_options
            )
        else:
            self.lr_scheduler = None

    def optimize(
        self,
        objective: Callable[[torch.tensor], torch.tensor],
        dataset: IterationBuffer,
    ) -> None:
        """Do gradient step.

        Args:
            objective (Callable[[torch.tensor], torch.tensor]): objective to optimize
            dataset (Dataset): data for optmization
        """

        dataloader = DataLoader(
            dataset=dataset,
            shuffle=self.shuffle,
            batch_size=len(dataset),
        )
        #print('optimize!', len(dataset.observations), len(dataset.actions), len(dataset.running_objectives), len(dataset.step_ids))
        batch_sample = next(iter(dataloader))
        self.optimizer.zero_grad()
        objective_value = objective(batch_sample)
        objective_value.backward()
        self.optimizer.step()

        last_observation = dataset.observations[-1]
        if self.lr_scheduler_switch(last_observation) and self.lr_scheduler is not None:
            self.lr_scheduler.step()