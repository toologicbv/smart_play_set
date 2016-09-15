from __future__ import print_function

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

# database called "data"
db = client.data
# A collection is a group of documents stored in MongoDB
collection = db.arduino

# usb_port = 'COM5'
# Ubuntu Linux USB port
usb_port= '/dev/ttyACM0'

arduinoSerialData = serial.Serial(usb_port, 9600) #Create Serial port object called arduinoSerialData
 
 
while True:
    minute = {"label": "movement", "data": {}}

    for s in range(60):
        one_second = []

        for i in range(40):
            if arduinoSerialData.inWaiting() > 0:
                data = str(arduinoSerialData.readline()[:-2])
                if data:
                    str_data = (''.join(data)).split(',')
                    if len(str_data) == 4:
                        print("split ", str_data[1:])
                        one_second.append(str_data[1:])
        print(one_second)
        minute["data"][str(s)] = one_second

    print(datetime.utcnow())
    collection.insert_one(minute)

