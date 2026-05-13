#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>

// ==========================================
// WiFi
// ==========================================

// const char* ssid = "Guest12345";
// const char* password = "loveorlando";

const char* ssid = "ESP32CAM_AP";
const char* password = "password123";
const char* hostname = "esp32cam_left";
// ==========================================
// TCP Server
// ==========================================

WiFiServer server(5050);

// ==========================================
// Camera Pins (AI Thinker)
// ==========================================

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==========================================
// Camera Setup
// ==========================================

bool initCamera() {

  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  config.pixel_format = PIXFORMAT_GRAYSCALE;

  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  return true;
}

void connectWiFi() {

  WiFi.setHostname(hostname);

  WiFi.begin(ssid, password);

  Serial.print("Connecting");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println(WiFi.localIP());

  if (MDNS.begin(hostname)) {
    Serial.println("mDNS started");
  } else {
    Serial.println("mDNS failed");
  }
    WiFi.setSleep(false);
}

// ==========================================
// Setup
// ==========================================

void setup() {

  Serial.begin(115200);

  if (!initCamera()) {
    return;
  }

  connectWiFi();

  server.begin();

  Serial.println("Camera server ready");
}

// ==========================================
// Loop
// ==========================================

void loop() {

  WiFiClient client = server.available();

    client.setNoDelay(true);

  if (!client) {
    delay(10);
    return;
  }

  Serial.println("Client connected");

  // Wait for command
  while (!client.available()) {
    delay(1);
  }

  String cmd = client.readStringUntil('\n');
  cmd.trim();

  if (cmd == "CAPTURE") {

    camera_fb_t* fb = esp_camera_fb_get();

    if (!fb) {
      Serial.println("Capture failed");
      client.stop();
      return;
    }

    uint32_t imageSize = fb->len;

    // Send size (big endian)
    uint8_t sizeBuffer[4];

    sizeBuffer[0] = (imageSize >> 24) & 0xFF;
    sizeBuffer[1] = (imageSize >> 16) & 0xFF;
    sizeBuffer[2] = (imageSize >> 8) & 0xFF;
    sizeBuffer[3] = imageSize & 0xFF;

    client.write(sizeBuffer, 4);

    // Send JPEG
    Serial.println("Writing");
    client.write(fb->buf, fb->len);
    Serial.println("Done Writing");
        

    Serial.printf("Sent %u bytes\n", fb->len);

    esp_camera_fb_return(fb);
  }

  client.stop();
}
