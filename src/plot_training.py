from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    monitor_csv = "artifacts/logs/monitor.csv"
    out_png = "artifacts/plots/reward_vs_episode.png"
    os.makedirs("artifacts/plots", exist_ok=True)

    df = pd.read_csv(monitor_csv, comment="#")
    rewards = df["r"].to_list()
    episodes = list(range(1, len(rewards) + 1))

    plt.figure()
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
