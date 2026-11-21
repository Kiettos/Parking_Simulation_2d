"""Callback huấn luyện cho ParkSim2D.

- RewardStatsCallback: cứ mỗi `log_every` step in mean/std reward (theo step)
  và mean/std episode return + success rate của các episode kết thúc trong cửa sổ.
- TargetReturnCallback: khi mean return của `window` episode gần nhất đạt
  `target_return`, chạy thêm `extra_episodes` episode rồi dừng train.
"""

import math

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardStatsCallback(BaseCallback):
    def __init__(self, log_every=10_000, verbose=1):
        super().__init__(verbose)
        self.log_every = max(1, int(log_every))
        self._next_log = self.log_every

        self._r_sum = 0.0
        self._r_sq = 0.0
        self._r_n = 0

        self._ep_returns = []
        self._ep_success = 0
        self._ep_count = 0

    def _on_step(self):
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64)
        if rewards.size:
            self._r_sum += float(rewards.sum())
            self._r_sq += float(np.square(rewards).sum())
            self._r_n += int(rewards.size)

        for info in self.locals.get("infos", []):
            event = info.get("decoy_event")
            if event == "spawn":
                print(
                    "[decoy] first success -> spawn decoys for next episode",
                    flush=True,
                )
            elif event == "respawn":
                print(
                    f"[decoy] layout {info.get('layout_index')} mastered:"
                    f" {info.get('completed_successes')} successes -> respawn decoys",
                    flush=True,
                )

            ep = info.get("episode")
            if ep is None:
                continue
            self._ep_returns.append(float(ep["r"]))
            self._ep_count += 1
            if info.get("is_success"):
                self._ep_success += 1

        if self.num_timesteps >= self._next_log:
            self._report()
            while self._next_log <= self.num_timesteps:
                self._next_log += self.log_every
        return True

    def _report(self):
        if self._r_n:
            step_mean = self._r_sum / self._r_n
            var = max(0.0, self._r_sq / self._r_n - step_mean ** 2)
            step_std = math.sqrt(var)
        else:
            step_mean = step_std = float("nan")

        if self._ep_returns:
            ep_mean = float(np.mean(self._ep_returns))
            ep_std = float(np.std(self._ep_returns))
            success = 100.0 * self._ep_success / self._ep_count
        else:
            ep_mean = ep_std = success = float("nan")

        print(
            f"[train] step={self.num_timesteps:>7d}"
            f" | reward/step mean={step_mean:+.3f} std={step_std:.3f}"
            f" | ep_return mean={ep_mean:+.1f} std={ep_std:.1f}"
            f" | success={success:.1f}%"
            f" ({self._ep_success}/{self._ep_count} episodes)",
            flush=True,
        )

        if self.logger is not None:
            self.logger.record("custom/reward_step_mean", step_mean)
            self.logger.record("custom/reward_step_std", step_std)
            if self._ep_returns:
                self.logger.record("custom/ep_return_mean", ep_mean)
                self.logger.record("custom/ep_return_std", ep_std)
                self.logger.record("custom/success_rate", success / 100.0)
            self.logger.dump(self.num_timesteps)

        self._r_sum = 0.0
        self._r_sq = 0.0
        self._r_n = 0
        self._ep_returns = []
        self._ep_success = 0
        self._ep_count = 0


class TargetReturnCallback(BaseCallback):
    """Dừng train sau khi đạt target và chạy thêm `extra_episodes` episode."""

    def __init__(
        self,
        target_return,
        window=100,
        extra_episodes=100,
        verbose=1,
    ):
        super().__init__(verbose)
        self.target_return = float(target_return)
        self.window = max(1, min(int(window), 100))  # ep_info_buffer maxlen = 100
        self.extra_episodes = max(0, int(extra_episodes))

        self.episodes_seen = 0
        self.hit_episode = None
        self.best_mean = float("-inf")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episodes_seen += 1

        buf = self.model.ep_info_buffer
        if buf is not None and len(buf) >= self.window:
            mean_return = float(np.mean([ep["r"] for ep in buf]))
            self.best_mean = max(self.best_mean, mean_return)
            if self.hit_episode is None and mean_return >= self.target_return:
                self.hit_episode = self.episodes_seen
                print(
                    f"[target] mean return {mean_return:.1f} >= {self.target_return:.1f}"
                    f" at episode {self.episodes_seen};"
                    f" running {self.extra_episodes} extra episodes...",
                    flush=True,
                )

        if (
            self.hit_episode is not None
            and self.episodes_seen >= self.hit_episode + self.extra_episodes
        ):
            print(
                f"[target] ran "
                f"{self.episodes_seen - self.hit_episode} extra episodes"
                f" (best mean return={self.best_mean:.1f}); stopping training.",
                flush=True,
            )
            return False
        return True
