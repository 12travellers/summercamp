from setuptools import setup, find_packages

print(find_packages())

setup(
    name='pymotionlib',
    version='0.1',
    package=find_packages(include=('pymotionlib',)),
    author='Libin Liu'
)