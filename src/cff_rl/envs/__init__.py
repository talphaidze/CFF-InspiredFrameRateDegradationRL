"""cff_rl.envs — registers project-local MiniWorld variants on import."""
from gymnasium.envs.registration import register

register(
    id="MiniWorld-FourRoomsHard-v0",
    entry_point="cff_rl.envs.fourrooms_hard:FourRoomsHard",
)

# Same as v0 but with the smaller (size=0.6) goal box, for ablation isolating
# the contribution of goal salience.
register(
    id="MiniWorld-FourRoomsHardSmallGoal-v0",
    entry_point="cff_rl.envs.fourrooms_hard:FourRoomsHard",
    kwargs={"goal_size": 0.6},
)

register(
    id="MiniWorld-FourRoomsHardDynamic-v0",
    entry_point="cff_rl.envs.fourrooms_hard_dynamic:FourRoomsHardDynamic",
    kwargs={"num_distractors": 4},
)

register(
    id="MiniWorld-FourRoomsHardDynamic2d-v0",
    entry_point="cff_rl.envs.fourrooms_hard_dynamic:FourRoomsHardDynamic",
    kwargs={"num_distractors": 2},
)
