import glob
import os
from setuptools import find_packages
from setuptools import setup

package_name = 'wall_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/'+package_name, ['package.xml', "wall_follower/params.yaml"]),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/wall_follower/launch', glob.glob(os.path.join('launch', '*launch.xml'))),
        ('share/wall_follower/launch', glob.glob(os.path.join('launch', '*launch.py')))],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='sebastianag2002@gmail.com',
    description='Wall Follower ROS2 Package',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wall_follower = wall_follower.wall_follower:main',
            'wall_follower_simulator = wall_follower.wall_follower_simulator:main',
	        'viz_example = wall_follower.viz_example:main',
        	'test_wall_follower = wall_follower.test_wall_follower:main',
            'stop_system = wall_follower.stop_system:main',
            'go_straight = wall_follower.go_straight:main',
            'step_test = wall_follower.step_test:main',
            'wall_follower_ryosei = wall_follower.wall_follower_ryosei:main',
        ],
    },
)
