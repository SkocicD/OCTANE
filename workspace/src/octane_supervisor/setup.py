from setuptools import find_packages, setup

package_name = 'octane_supervisor'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['octane_supervisor/config/faults.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adam Carbone',
    maintainer_email='adam.r.carbone@live.com',
    description='State machine and fault management for the CSU Lunabotics rover.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mode_manager_node = octane_supervisor.nodes.mode_manager_node:main',
            'fault_manager_node = octane_supervisor.nodes.fault_manager_node:main',
            'fault_checker_node = octane_supervisor.nodes.fault_checker_node:main',
        'state_monitor_node = octane_supervisor.nodes.state_monitor_node:main',
        ],
    },
)
