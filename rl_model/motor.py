from typing import Set

import Mock.GPIO as GPIO


class FourWheelMotor:
    def __init__(self, speed: int, pins: Set[int] = set([11, 12, 13, 15])):
        GPIO.setmode(GPIO.BOARD)
        self.speed = speed
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            self.pins = pins.add(pin)

    def set_speed(self, speed):
        self.speed = speed

    def get_speed(self):
        return self.speed

    def get_pins(self):
        return self.pins

    def __str__(self):
        return f"Motor(speed={self.speed}, pins={self.pins})"

    def move_forward(self):
        print("Moving forward")

    def move_backward(self):
        print("Moving backward")

    def stop(self):
        print("Stopping")

    def turn_left(self):
        print("Turning left")

    def turn_right(self):
        print("Turning right")
