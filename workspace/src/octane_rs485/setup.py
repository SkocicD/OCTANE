from setuptools import find_packages, setup

package_name = 'octane_rs485'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Adam Carbone',
    maintainer_email='adam.r.carbone@live.com',
    description='RS485 Modbus RTU driver for BLD-510B replacement motor.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rs485_drive_node  = octane_rs485.nodes.rs485_drive_node:main',
            'rs485_debug_node  = octane_rs485.nodes.rs485_debug_node:main',
        ],
    },
)
