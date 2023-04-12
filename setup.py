from setuptools import setup, find_packages
import subprocess
import os
import sys
import logging
from rich import print

dir_path = os.path.dirname(__file__)

# requirements
requirements_file = os.path.join(dir_path, 'requirements.txt')
install_requires = [l.strip() for l in open(requirements_file).readlines()]

# third
l = f"cd {dir_path} && cd third/pyapollo && pip install ."
os.system(l)

# setup
setup(
    name="redpy",
    version="0.0.1",
    author="wangqixun",
    author_email="253817124@qq.com",

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=[],
)


