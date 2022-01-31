from setuptools import setup

setup(
    name='gym_snake',
    version='0.1.0',
    author="Jesse Chen",
    description='Set of snake environments for OpenAI Gym',
    install_requires=[
        'gym',
        'numpy',
        'matplotlib'
    ],
    extras_require={
        'imitation_scripts': ["imitation", "stable_baselines3", "keyboard"]
    }
)