string serial
sensor_msgs/Image image
sensor_msgs/CameraInfo info
octane_msgs/CameraParam param
octane_msgs/CameraOffset offset

# EXAMPLE USAGE:
#   msg.serial                       — camera serial number
#   msg.image                        — raw sensor_msgs/Image
#   msg.info                         — sensor_msgs/CameraInfo
#   msg.param.**attribute**          — camera parameters (.fx, .fy, .cx, .cy, .dist, .width, .height)
#   msg.offset.pos.**attribute**     — mounting position in m relative to robot base (.x, .y, .z)
#   msg.offset.rot.**attribute**     — mounting rotation in deg (Euler) relative to robot base (.x, .y, .z)
