import argparse
import random

from hexfall.env import HexFallEnv


def run(
    level_path: str,
    env_seed: int | None = None,
    agent_seed: int | None = None,
    max_steps: int = 10_000,
) -> None:
    env = HexFallEnv(level_path, seed=env_seed)
    obs, _ = env.reset()
    rng = random.Random(agent_seed)
    for step in range(max_steps):
        legal = [i for i, m in enumerate(obs["action_mask"]) if m]
        if not legal:
            print(f"No legal actions at step {step}.")
            return
        action = rng.choice(legal)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(
                f"Done in {step + 1} steps. "
                f"Reason: {info['termination_reason']}, reward: {reward}"
            )
            return
    print(f"Hit max_steps={max_steps} without termination.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("level_path")
    p.add_argument("--env-seed", type=int, default=None)
    p.add_argument("--agent-seed", type=int, default=None)
    args = p.parse_args()
    run(args.level_path, args.env_seed, args.agent_seed)
