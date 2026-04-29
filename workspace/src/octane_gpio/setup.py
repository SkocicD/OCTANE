from setuptools import find_packages, setup

package_name = 'octane_gpio'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adam Carbone',
    maintainer_email='adam.r.carbone@live.com',
    description='GPIO hardware interface for OCTANE actuator relays.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gpio_actuator_node = octane_gpio.nodes.gpio_actuator_node:main',
        ],
    },
)
