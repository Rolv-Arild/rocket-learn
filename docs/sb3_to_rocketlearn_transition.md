# SWITCHING FROM [STABLE-BASELINES3](https://stable-baselines3.readthedocs.io/en/master/) TO ROCKET-LEARN

When you install rocket-learn, verify that the version of [rlgym](https://rlgym.org/) is compatible with rocket-learn. Use only the release version of rlgym. Do not use beta of rlgym unless you are attempting to beta test. When setting up your environment, make sure you do not install rlgym from a cached version on your machine, and verify that the dll in the [bakkesmod](https://bakkesmod.com/) plugin folder is accurate.

SB3 abstracts away several important parts of ML training that rocket-learn does not.

- Your rewards will not be normalized
- Your networks will not have orthogonal initialization by default (assuming you use PPO)

This can drastically affect the results you get and it is not uncommon to not see the same results
in rocket-learn as you did in SB3, at least until you make tweaks. In addition to the major
differences listed above, differences in implementation in learning algorithms can cause large
changes in results. Be prepared to do some extra tweaking as a part of the switch.
