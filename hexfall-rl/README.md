# hexfall-rl

Reinforcement-learning simulator for Hex Fall (Paxie Games). Implements the game as a partially observable MDP: a hex-stack field is cleared by selecting colored buckets from a reserve into a 5-slot buffer. This repo provides the core data types, level loader, and (in later sessions) the simulator and Gymnasium environment used for RL training and automated difficulty assessment.
