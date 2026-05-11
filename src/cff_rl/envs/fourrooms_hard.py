"""FourRoomsHard: vanilla FourRooms + fixed pillar obstacles + shape-distractor balls.

Same 4-room shell and connection topology as `MiniWorld-FourRooms-v0`, plus:
- Four fixed grey pillar boxes (one per room) that act as static obstacles.
- N ball distractors (default 3) of varied colors, randomly placed.

Goal is unchanged: navigate to the red box. Distractors are *Balls* rather than
non-red Boxes because the agent observes 64x64 grayscale — color alone wouldn't
discriminate; shape silhouette (sphere vs cube) does.
"""
from gymnasium import spaces, utils

from miniworld.entity import Ball, Box, COLOR_NAMES
from miniworld.miniworld import MiniWorldEnv


class FourRoomsHard(MiniWorldEnv, utils.EzPickle):
    """FourRooms with shape-distractor balls and fixed pillar obstacles."""

    def __init__(
        self,
        num_distractors: int = 1,
        pillar_size: float = 1.0,
        goal_size: float = 0.9,
        **kwargs,
    ):
        self.num_distractors = int(num_distractors)
        self.pillar_size = float(pillar_size)
        self.goal_size = float(goal_size)
        MiniWorldEnv.__init__(self, max_episode_steps=1000, **kwargs)
        utils.EzPickle.__init__(
            self,
            num_distractors=num_distractors,
            pillar_size=pillar_size,
            goal_size=goal_size,
            **kwargs,
        )

        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        room0 = self.add_rect_room(min_x=-7, max_x=-1, min_z=1, max_z=7)
        room1 = self.add_rect_room(min_x=1, max_x=7, min_z=1, max_z=7)
        room2 = self.add_rect_room(min_x=1, max_x=7, min_z=-7, max_z=-1)
        room3 = self.add_rect_room(min_x=-7, max_x=-1, min_z=-7, max_z=-1)

        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        # Pillars: full-height columns so they read as wall geometry (not
        # objects). MiniWorld ceilings are ~y=2.7; a height of 3 reaches the
        # ceiling and is visually unambiguous as wall in grayscale.
        pillar_dims = [self.pillar_size, 3.0, self.pillar_size]
        for px, pz in [(-4, 4), (4, 4), (4, -4), (-4, -4)]:
            self.place_entity(
                Box(color="grey", size=pillar_dims), pos=(px, 0, pz)
            )

        self.box = self.place_entity(Box(color="red", size=self.goal_size))

        distractor_colors = [c for c in COLOR_NAMES if c not in ("red", "grey")]
        for i in range(self.num_distractors):
            color = distractor_colors[i % len(distractor_colors)]
            self.place_entity(Ball(color=color, size=0.6))

        self.place_agent()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.near(self.box):
            reward += self._reward()
            terminated = True
        return obs, reward, terminated, truncated, info
