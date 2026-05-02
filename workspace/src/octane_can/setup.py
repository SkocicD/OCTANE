from setuptools import find_packages, setup

package_name = 'octane_can'

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
    description='CAN bus hardware interface for OCTANE drive motors.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'can_drive_node = octane_can.nodes.can_drive_node:main',
        ],
    },
)
