#!/usr/bin/env python3
import time, json, random
import paho.mqtt.client as mqtt

BROKER   = "broker.emqx.io"
PORT     = 1883
TOPIC    = "drill/data"

pub = mqtt.Client()
pub.connect(BROKER, PORT, 60)

# send 300 tilfeldige RMS-verdier mellom 0.1 og 1.2
for _ in range(300):
    rms = random.uniform(0.1, 1.2)
    msg = json.dumps({"rms": rms})
    pub.publish(TOPIC, msg)
    print("Published", msg)
    time.sleep(0.005)

pub.disconnect()
