from setuptools import find_packages, setup

package_name = 'octane_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Adam Carbone',
    maintainer_email='adam.r.carbone@live.com',
    description='Perception stack for the CSU Lunabotics rover.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'astra_depth_node = octane_perception.nodes.astra_depth_node:main',
            'rgb_camera_node = octane_perception.nodes.rgb_camera_node:main',
            'depth_estimation_node = octane_perception.nodes.depth_estimation_node:main',
        ],
    },
)
