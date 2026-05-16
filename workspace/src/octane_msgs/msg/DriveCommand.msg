# Tank drive velocity command (normalized)
# Values: -1.0 (full reverse) to 1.0 (full forward)
# Published by manual_drive_node, subscribed by CAN hardware node
float32 left_velocity
float32 right_velocity

# GUI speed dial — multiplier as integer percentage (0-500).
# 100 = 1.0x (normal). Default 100 preserves existing behaviour when field is unset.
# Effective speed = velocity * speed_scale * (speed_modifier / 100.0)
# clamped to [0.0, 1.0] before motor output.
uint16 speed_modifier 100
