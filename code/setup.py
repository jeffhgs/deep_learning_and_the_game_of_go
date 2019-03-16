from setuptools import setup
from setuptools import find_packages

setup(name='dlgo',
      version='0.2',
      description='Deep Learning and the Game of Go',
      url='http://github.com/maxpumperla/deep_learning_and_the_game_of_go',
      install_requires=[
            'numpy', 
            'tensorflow', 
            'keras', 
            'gomill', 
            'Flask', 
            'Flask-Cors', 
            'future', 
            'h5py', 
            'six'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
