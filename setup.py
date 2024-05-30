from setuptools import setup, find_packages

setup(
   name='hspn',
   version='0.0.1',
   description='deep operator networks for ARL CFD',
   author='Jasmine Ratchford',
   author_email='jratchford@sei.cmu.edu',
   package_dir={"": "src"},
   packages=find_packages(where="src", include=["src.hspn*"]), 
   install_requires=['matplotlib', 'numpy', 'horovod'], #external packages as dependencies
)