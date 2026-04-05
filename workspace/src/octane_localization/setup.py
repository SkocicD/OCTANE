from setuptools import setup, find_packages

package_name = 'octane_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='skocicdavid@gmail.com',
    description='AprilTag detector node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'april_tag_detector = octane_localization.nodes.april_tag_detector:main',
        ],
    },
)
