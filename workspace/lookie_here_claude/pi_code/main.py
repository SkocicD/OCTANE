import socket
import struct
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# Camera Hostnames
# ==========================================

CAMERAS = [
    "esp32cam_left.local",
    # "esp32cam_right.local",
]

PORT = 5050

SAVE_DIR = "photos"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# Receive Exact Bytes
# ==========================================


def recv_exact(sock, size):

    data = b""

    while len(data) < size:

        packet = sock.recv(size - len(data))

        if not packet:
            return None

        data += packet

    return data

# ==========================================
# Fetch Image From Camera
# ==========================================


def fetch_camera(hostname):

    camera_name = hostname.replace(".local", "")

    print(f"[+] Connecting to {camera_name}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:

        sock.settimeout(10)

        sock.connect((hostname, PORT))

        # Send capture command
        sock.sendall(b"CAPTURE\n")

        # ----------------------------------
        # Receive image size
        # ----------------------------------

        raw_size = recv_exact(sock, 4)

        if not raw_size:
            print(f"[-] {camera_name}: failed to receive size")
            return

        image_size = struct.unpack(">I", raw_size)[0]

        print(f"[+] {camera_name}: image size = {image_size} bytes")

        # ----------------------------------
        # Receive image data
        # ----------------------------------

        image_data = recv_exact(sock, image_size)

        if not image_data:
            print(f"[-] {camera_name}: failed image receive")
            return

        # ----------------------------------
        # Save image
        # ----------------------------------

        camera_dir = os.path.join(SAVE_DIR, camera_name)

        os.makedirs(camera_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = os.path.join(
            camera_dir,
            f"{timestamp}.jpg"
        )

        with open(filename, "wb") as f:
            f.write(image_data)

        print(f"[+] Saved {filename}")

    except Exception as e:
        print(f"[-] {camera_name}: {e}")

    finally:
        sock.close()

# ==========================================
# Main Loop
# ==========================================


def main():

    print("[+] ESP32-CAM Controller Started")

    while True:

        # Poll all cameras in parallel
        with ThreadPoolExecutor(max_workers=len(CAMERAS)) as executor:

            executor.map(fetch_camera, CAMERAS)

        # Wait before next capture round
        time.sleep(10)

# ==========================================
# Start
# ==========================================


if __name__ == "__main__":
    main()
