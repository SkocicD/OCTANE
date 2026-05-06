from setuptools import find_packages, setup

package_name = 'octane_sensors'

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
    description='Hardware sensor drivers for the OCTANE rover.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'adxl345_node = octane_sensors.nodes.adxl345_node:main',
        ],
    },
)
