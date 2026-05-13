from setuptools import find_packages, setup

package_name = 'octane_manual_ctrl'

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
    description='Manual control nodes for OCTANE rover.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manual_drive_node = octane_manual_ctrl.nodes.manual_drive_node:main',
            'manual_actuator_node = octane_manual_ctrl.nodes.manual_actuator_node:main',
            'can_drive_node = octane_manual_ctrl.nodes.can_drive_node:main',
            'can_debug_node = octane_manual_ctrl.nodes.can_debug_node:main',
            'actuator_debug_node = octane_manual_ctrl.nodes.actuator_debug_node:main',
        ],
    },
)
