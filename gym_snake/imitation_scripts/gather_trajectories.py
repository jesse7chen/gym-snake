import os
import os.path as osp
from typing import Mapping, Optional

import sacred.run
from sacred.observers import FileStorageObserver

from imitation.data import rollout, wrappers
from imitation.scripts.common import common
from gym_snake.imitation_scripts.gather_trajectories_config import snake_ex


@snake_ex.main
def gather_trajectories(
    *,
    _run: sacred.run.Run,
    _seed: int,
    rollout_save_n_timesteps: Optional[int],
    rollout_save_n_episodes: Optional[int],
    agent
) -> Mapping[str, float]:
    """Saves Snake game trajectories obtained from either human play or from
    play generated by a Hamiltonian cycle-based agent

        - Rollouts are saved to `{log_dir}/rollouts/{step}.pkl`.

    Args:
        rollout_save_n_timesteps: The minimum number of timesteps saved in every
            file. Could be more than `rollout_save_n_timesteps` because
            trajectories are saved by episode rather than by transition.
            Must set exactly one of `rollout_save_n_timesteps`
            and `rollout_save_n_episodes`.
        rollout_save_n_episodes: The number of episodes saved in every
            file. Must set exactly one of `rollout_save_n_timesteps` and
            `rollout_save_n_episodes`.

    Returns:
        The return value of `rollout_stats()` using the final policy.
    """
    custom_logger, log_dir = common.setup_logging()
    rollout_dir = osp.join(log_dir, "rollouts")
    os.makedirs(rollout_dir, exist_ok=True)

    venv = common.make_venv(
        post_wrappers=[lambda env, idx: wrappers.RolloutInfoWrapper(env)],
    )

    # Generate and save trajectories
    save_path = osp.join(rollout_dir, "final.pkl")
    sample_until = rollout.make_sample_until(
        rollout_save_n_timesteps,
        rollout_save_n_episodes,
    )

    rollout.rollout_and_save(save_path, agent, venv, sample_until)


def main_console():
    observer = FileStorageObserver(osp.join("../agents/output", "sacred", "snake_ex"))
    snake_ex.observers.append(observer)
    snake_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
