from setuptools import setup

long_description = """

# Rocket Learn

Learning!

"""

setup(
   name='rocket_learn',
   version='0.1',
   description='Rocket Learn',
   author=['Rolv Arild', 'Daniel Downs'],
   url='https://github.com/Rolv-Arild/rocket-learn',
   packages=['rocket_learn', 'rocket_learn.agent', 'rocket_learn.rollout_generator', 'rocket_learn.utils'], 
   long_description=long_description,
   install_requires=['gym', 'torch', 'tqdm', 'trueskill', 'msgpack_numpy', 'wandb'],
)