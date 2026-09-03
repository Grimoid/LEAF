from setuptools import find_packages, setup


def _requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="leaf-speech-rl",
    version="1.0.0",
    description="LEAF: prefix-tree reinforcement learning for speech-aware LLMs",
    author="Argyrios Gerogiannis",
    packages=find_packages(include=["openrlhf", "openrlhf.*"]),
    install_requires=_requirements(),
    python_requires=">=3.10",
)
