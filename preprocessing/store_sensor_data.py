from datetime import datetime
from pymongo import MongoClient
import serial

# this script requires:
# - an arduino connected to the serial port (may be different from COM5)
# - a running instance of mongod (MongoDB server)
#
# One document contains a minute of data, where samples are collected per second
# For the current prototype the timing of the serial port is a bit of, the 'seconds'
# actually take a bit longer.

client = MongoClient() 
db = client.data
collection = db.arduino

arduino = serial.Serial('COM5', 9600)

while True:
  minute = {"label": "movement", "data": {}}

  for s in range(60):
    one_second = []

    for i in range(40):
      data = str(arduino.readline()[:-2], 'utf-8')
      if data:
        sensor_values = [d for d in data.split()]
        
        one_second.append(sensor_values)

    minute["data"][str(s)] = one_second
  
  print(datetime.utcnow())
  collection.insert_one(minute)