from setuptools import setup, find_packages

long_description = """

# Rocket Learn

Learning!

"""

setup(
   name='rocket_learn',
   version='0.1.5',
   description='Rocket Learn',
   author=['Rolv Arild', 'Daniel Downs'],
   url='https://github.com/Rolv-Arild/rocket-learn',
   packages=[package for package in find_packages() if package.startswith("rocket_learn")],
   long_description=long_description,
   install_requires=['cloudpickle==1.6.0', 'gym', 'torch', 'tqdm', 'trueskill', 'msgpack_numpy', 'wandb'],
)
