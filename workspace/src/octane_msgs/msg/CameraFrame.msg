string serial
sensor_msgs/Image image
sensor_msgs/CameraInfo info
octane_msgs/CameraParam param

# EXAMPLE USAGE:
#   msg.serial                — camera serial number
#   msg.image                 — raw sensor_msgs/Image
#   msg.info                  — sensor_msgs/CameraInfo
#   msg.param.**attribute**   — camera parameters (.fx, .fy, .cx, .cy, .dist, .width, .height)      
