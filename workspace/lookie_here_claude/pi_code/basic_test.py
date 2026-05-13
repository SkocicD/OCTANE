#!/usr/bin/env python3

import socket
import struct

HOST = "esp32cam_left.local"
PORT = 5050

# connect to ESP32-CAM
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect((HOST, PORT))

print("connected")

# request capture
sock.sendall(b"CAPTURE\n")

# receive image size
raw_size = sock.recv(4)

if len(raw_size) != 4:
    raise Exception("failed to receive size")

image_size = struct.unpack(">I", raw_size)[0]

print("image size:", image_size)

# receive image
received = 0

with open("test.jpg", "wb") as f:

    while received < image_size:

        chunk = sock.recv(4096)

        if not chunk:
            break

        f.write(chunk)

        received += len(chunk)

print("received:", received)

sock.close()
