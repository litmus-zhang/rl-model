import Mock.GPIO as GPIO


class Sensor:
    def __init__(self, pins: int):
        GPIO.setmode(GPIO.BOARD)
        self.pins = pins
        for pin in pins:
            GPIO.setup(pin, GPIO.IN)
            self.pins = pins.add(pin)

    def get_pins(self):
        return self.pins

    def __str__(self):
        return f"Sensor(pins={self.pins})"

    def read(self):
        print("Reading sensor")
        return 0
