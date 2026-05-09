from setuptools import setup, find_packages

setup(
    name="octane_terrain_mapping",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "inference_node = octane_terrain_mapping.inference_node:main",
        ],
    },
    install_requires=["numpy", "torch", "sensor-msgs-py", "scikit-learn", "scipy"],
)
