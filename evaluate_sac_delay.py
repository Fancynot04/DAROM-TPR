"""Evaluate an Antwerp checkpoint in a process separate from training."""

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import SAC

from src.merging import Merging
from src.wrapper import DelayWrapper, SafetyWrapper


OUTCOMES = ('Arrived', 'Collision', 'Max Steps', 'Emergency Stopping')


def evaluate(model, env, episodes, seed):
    counts = Counter()
    returns = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                counts[info.get('message', 'Unknown')] += 1
                returns.append(episode_return)
                break
    return {**{outcome: counts[outcome] for outcome in OUTCOMES},
            'mean_return': float(np.mean(returns))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Antwerp SAC checkpoint (.zip)')
    parser.add_argument('--scenario', choices=('easy', 'medium', 'hard'), default='medium')
    parser.add_argument('--delay-mode', choices=('uniform', 'exponential', 'triangular', 'bursty', 'bimodal'), default='uniform')
    parser.add_argument('--episodes', type=int, default=51)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--safety', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')

    env = DelayWrapper(Merging(seed=args.seed, scenario=args.scenario),
                       max_delay=20, mode='all', delay_mode=args.delay_mode,
                       seed=args.seed)
    if args.safety:
        env = SafetyWrapper(env)
    try:
        model = SAC.load(args.checkpoint, device='auto')
        result = evaluate(model, env, args.episodes, args.seed)
    finally:
        env.close()

    print(f'{args.scenario} | {args.delay_mode} | safety={args.safety} | episodes={args.episodes}')
    for outcome in OUTCOMES:
        print(f'{outcome}: {result[outcome]}/{args.episodes} ({result[outcome] / args.episodes:.1%})')
    print(f"Mean return: {result['mean_return']:.2f}")


if __name__ == '__main__':
    main()
