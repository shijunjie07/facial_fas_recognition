# --------------------------
# setup env
# @author: Shi Junjie
# Wed 13 June 2024
# --------------------------

import os
from setuptools import setup, find_packages

with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name = "facial_fas_recognition",
    version = '1.0.0',
    description='The Facial Recognition with Anti-Spoofing is a system \
        which capable of accurately identifying individuals while simultaneously \
            detecting and preventing spoofing attempts. Spoofing attempts include \
                using photographs, videos, or masks to deceive the facial recognition system.',
    
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Shi Junjie',
    author_email='shijunja@gmail.com',
    url='https://github.com/shijunjie07/facial_fas_recognition.git',  # Project URL
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.12',
)