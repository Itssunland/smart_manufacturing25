//connecting via MQTT protocoll
#include <WiFi.h>
#include <Wire.h>
#include <PubSubClient.h>
#include "MPU6050.h"
#include <cmath>

#define MPU6050_I2C_ADDRESS 0x68
#define MPU6050_WHO_AM_I     0x75

const char* SSID_SEC = "ABC";
const char* PASS_SEC = "99999999";
const char* MQTT_BROKER = "broker.emqx.io";
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "drill/data";

WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

MPU6050 imu;
const int WINDOW_SIZE = 256;
float buf[WINDOW_SIZE];
int   idx = 0;

float computeRMS() {
  float sum = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) sum += buf[i] * buf[i];
  return sqrt(sum / WINDOW_SIZE);
}

void connectSecureWiFi() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  delay(100);
  WiFi.begin(SSID_SEC, PASS_SEC);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > 15000) return;
    delay(500);
  }
}

void reconnectMQTT() {
  static char clientId[32];
  snprintf(clientId, sizeof(clientId), "esp32-%lu", millis());
  while (!mqtt.connected()) {
    mqtt.connect(clientId);
    delay(500);
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  connectSecureWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  Wire.begin();
  imu.initialize();
  if (!imu.testConnection()) while (1) delay(1000);
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqtt.connected()) reconnectMQTT();
    mqtt.loop();
    int16_t ax, ay, az;
    imu.getAcceleration(&ax, &ay, &az);
    float vib = sqrt(
      sq(ax / 16384.0 * 9.81F) +
      sq(ay / 16384.0 * 9.81F) +
      sq(az / 16384.0 * 9.81F)
    );
    buf[idx++] = vib;
    if (idx >= WINDOW_SIZE) {
      idx = 0;
      float rms = computeRMS();
      char msg[64];
      snprintf(msg, sizeof(msg), "{\"rms\":%.3f}", rms);
      mqtt.publish(MQTT_TOPIC, msg);
    }
  } else {
    connectSecureWiFi();
  }
  delay(1);
}
