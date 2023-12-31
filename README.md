# RL PROJECT - CartPole swing up
## Authors: Bogdan Alexandrov, Vadim Shirokinskiy

### All code can be ran in RL_project.ipynb

Just run cells.

*system* dir contains code for dynamic system.

*buffer* dir contains code for storing data and statistics from simulated episode.

*models* dir contains models to train.

*policy* dir contains policy to affect the system

*simulator* dir contains code for monte carlo simulations

### To run simulation you need to create next lines of code:

system = InvertedPendulumSystem(...)

simulator = Simulator(...)

model = DiscreteModel(...)

optimizer = Optimizer(...)

policy = PolicyREINFORCE(...)

trivial_termination_criterion = lambda *args: False

scenario = MonteCarloSimulationScenario(<br />
    simulator=simulator,<br />
    system=system, <br />
    policy=policy, <br />
    N_episodes=2, - num of episodes <br />
    N_iterations=200, - num of iterations <br />
    termination_criterion=trivial_termination_criterion, <br />
    discount_factor=1.0, - gamma <br />
)
