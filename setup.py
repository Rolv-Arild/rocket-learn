from setuptools import setup, find_packages

long_description = """

# Rocket Learn

Learning!

"""

setup(
   name='rocket_learn',
   version='1.0.0a0',
   description='Rocket Learn',
   author='Rolv-Arild Braaten, Daniel Downs',
   url='https://github.com/Rolv-Arild/rocket-learn',
   packages=[package for package in find_packages() if package.startswith("rocket_learn")],
   long_description=long_description,
   install_requires=[
      'rlgym-rocket-league[sim]>=2.0.0rc0',
      'rlgym-api>=2.0.0rc0',
      'rlgym-tools>=2.0.0rc0',
      'torch',
      'torchrl',
      'tensordict'
   ],
)
