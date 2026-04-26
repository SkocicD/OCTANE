from setuptools import find_packages, setup

package_name = 'octane_network'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['octane_network/config/network_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adam Carbone',
    maintainer_email='adam.r.carbone@live.com',
    description='Wireless communication for OCTANE rover ground station link.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'network_comm_node = octane_network.nodes.network_comm_node:main',
            'heartbeat_sender = octane_network.nodes.heartbeat_sender:main',
        ],
    },
)
