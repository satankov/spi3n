from setuptools import setup, find_packages

with open("requirements.txt", 'r') as f:
    requirements = f.read()

setup(
    name='spi3n',
    version='0.9',
    description='Spin NN module',
    author='Satankov',
    author_email='satankow@yandex.ru',
    packages=find_packages(),
    install_requires=requirements.split('\n')
)
