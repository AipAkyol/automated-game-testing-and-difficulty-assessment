import argparse
import random

from hexfall.env import HexFallEnv
from hexfall.render import render


def run(
    level_path: str,
    env_seed: int | None = None,
    agent_seed: int | None = None,
    max_steps: int = 10_000,
    mode: str = "agent",
) -> None:
    env = HexFallEnv(level_path, seed=env_seed)
    obs, _ = env.reset()
    cols = env._reserve_cols

    print(f"=== INITIAL STATE (mode={mode}) ===")
    print("Last action: (none -- initial state)")
    print(render(env, mode))

    rng = random.Random(agent_seed)
    for step in range(max_steps):
        legal = [i for i, m in enumerate(obs["action_mask"]) if m]
        if not legal:
            print(f"No legal actions at step {step}.")
            return
        action = rng.choice(legal)
        row, col = action // cols, action % cols
        picked = obs["reserve"][row][col]
        picked_summary = (
            f"Last action: picked {picked.get('type', '?')}:"
            f"{picked.get('color', '?')} @ reserve[{row},{col}]"
        )

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"\n=== STEP {step + 1} | action={action} reward={reward} "
            f"terminated={terminated} info={info} ==="
        )
        print(picked_summary)
        print(render(env, mode))

        if terminated or truncated:
            print(
                f"\nDone in {step + 1} steps. "
                f"Reason: {info['termination_reason']}, reward: {reward}"
            )
            return
    print(f"Hit max_steps={max_steps} without termination.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--level", dest="level_path", required=True)
    p.add_argument("--env-seed", type=int, default=None)
    p.add_argument("--agent-seed", type=int, default=None)
    p.add_argument(
        "--mode",
        choices=("agent", "full"),
        default="agent",
        help="agent = observation-limited view; full = debug view with all hidden state",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Shortcut for --mode full",
    )
    args = p.parse_args(argv)
    mode = "full" if args.full else args.mode
    run(args.level_path, args.env_seed, args.agent_seed, mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
