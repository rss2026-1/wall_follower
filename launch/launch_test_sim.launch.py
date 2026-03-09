from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()
    
    building_yaml = os.path.join(
        get_package_share_directory('racecar_simulator'),
        'maps',
        'building_31.yaml'
    )
     
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name='map_server',
        output="screen",
        parameters=[{
            "yaml_filename": building_yaml}]
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name='lifecycle_manager',
        output="screen",
        parameters=[{
            "autostart": True,
            "node_names": ['map_server']}]
    )

    racecar_model = IncludeLaunchDescription(
        XMLLaunchDescriptionSource([os.path.join(
            get_package_share_directory('racecar_simulator'), 'launch'),
            '/racecar_model.launch.xml'])
    )
    
    config = os.path.join(
        get_package_share_directory('racecar_simulator'),
        'params.yaml'
        )
    
    racecar_simulator = Node(
        package="racecar_simulator",
        executable="simulate",
        name='racecar_simulator',
        output="screen",
        parameters=[config]
    )

    wall_follower_simulator = Node(
        package="wall_follower",
        executable="wall_follower_simulator",
        name='wall_follower_simulator',
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "drive_topic": "/drive",
            "velocity": 1.0,
            "desired_distance": 1.0,
            "side": -1,
        }]
    )

    
    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)
    ld.add_action(racecar_model)
    ld.add_action(racecar_simulator)
    ld.add_action(wall_follower_simulator)
    

    return ld