# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:46:58 2025

@author: Pulee
"""

import serial
import time

arduino_port = 'COM9'
baud_rate = 9600

try:
    arduino = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)
    print("Serial port connection successful!")

    arduino.write(b'T')
    print("Sent:T(TTL ON)")
    time.sleep(10)

    arduino.write(b't')
    print("Sent:t(TTL OFF)")
    time.sleep(10)

    arduino.write(b'L')
    print("Sent:L(Laser Pulse)")

    # for _ in range(5):
    #     arduino.write(b'T')
    #     time.sleep(0.3)
    #     arduino.write(b't')
    #     time.sleep(0.3)

    arduino.close()
    print("Test completed, serial port closed.")

except Exception as e:
    print(f"Serial communication failed:{e}")
