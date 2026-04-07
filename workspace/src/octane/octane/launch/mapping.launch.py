from launch import LaunchDescription
from launch.actions import LogInfo


# Generate launch description for mapping subsystem
def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='Starting mapping subsystem'),
        # TODO: add mapping nodes as they are implemented
        LogInfo(msg='Mapping subsystem online'),
    ])
