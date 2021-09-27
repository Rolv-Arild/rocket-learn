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
   packages=['rocket_learn'], 
   long_description=long_description,
   install_requires=['gym', 'rlgym', 'torch', 'tqdm', 'trueskill', 'msgpack_numpy', 'wandb'],
)