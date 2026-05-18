"""FourRoomsHardDynamic: FourRoomsHard with bouncing distractor balls.

Layout (rooms, pillars, goal, ball placement) is identical to FourRoomsHard.
Distractor balls move at constant speed in the xz-plane and reflect off walls,
pillars, the agent, and each other. Speed is a single fixed knob per run; sweep
across speeds externally.
"""
import numpy as np
from gymnasium import utils

from miniworld.entity import Ball

from cff_rl.envs.fourrooms_hard import FourRoomsHard


class FourRoomsHardDynamic(FourRoomsHard):
    """FourRoomsHard with constant-velocity bouncing distractor balls."""

    def __init__(
        self,
        distractor_speed: float = 0.1,
        distance_reward: float = 0.0,
        **kwargs,
    ):
        self.distractor_speed = float(distractor_speed)
        # Dense reward shaping: per-step reward of distance_reward * (d_prev -
        # d_curr) in Euclidean distance to the goal (xz-plane). Encourages
        # closing distance without overriding sparse goal reward.
        self.distance_reward = float(distance_reward)
        self._distractors: list = []
        self._distractor_vels: list = []
        super().__init__(**kwargs)
        utils.EzPickle.__init__(
            self,
            distractor_speed=distractor_speed,
            distance_reward=distance_reward,
            **kwargs,
        )

    def _gen_world(self):
        super()._gen_world()
        self._distractors = [
            e for e in self.entities if isinstance(e, Ball) and e is not self.box
        ]
        self._distractor_vels = []
        for _ in self._distractors:
            theta = self.np_random.uniform(0, 2 * np.pi)
            self._distractor_vels.append(
                np.array(
                    [
                        self.distractor_speed * np.cos(theta),
                        self.distractor_speed * np.sin(theta),
                    ],
                    dtype=np.float32,
                )
            )

    def _move_distractors(self):
        for ball, vel in zip(self._distractors, self._distractor_vels):
            vx, vz = float(vel[0]), float(vel[1])
            base = ball.pos

            new_pos = base + np.array([vx, 0.0, vz], dtype=base.dtype)
            if not self.intersect(ball, new_pos, ball.radius):
                ball.pos = new_pos
                continue

            # Axis-separated swept collision: try x and z independently and
            # reflect the blocked component(s).
            try_x = base + np.array([vx, 0.0, 0.0], dtype=base.dtype)
            try_z = base + np.array([0.0, 0.0, vz], dtype=base.dtype)
            x_blocked = bool(self.intersect(ball, try_x, ball.radius))
            z_blocked = bool(self.intersect(ball, try_z, ball.radius))

            if x_blocked:
                vel[0] = -vel[0]
            if z_blocked:
                vel[1] = -vel[1]

            if not x_blocked:
                ball.pos = try_x
            elif not z_blocked:
                ball.pos = try_z
            # else: both blocked — flipped both components, stay put this step.

    def step(self, action):
        self._move_distractors()
        d_before = (
            float(np.linalg.norm((self.agent.pos - self.box.pos)[[0, 2]]))
            if self.distance_reward > 0
            else 0.0
        )
        obs, reward, terminated, truncated, info = super().step(action)
        if self.distance_reward > 0:
            d_after = float(np.linalg.norm((self.agent.pos - self.box.pos)[[0, 2]]))
            reward += self.distance_reward * (d_before - d_after)
        return obs, reward, terminated, truncated, info
