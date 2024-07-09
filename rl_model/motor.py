from typing import List

import Mock.GPIO as GPIO

# import RPi.GPIO as GPIO


class FourWheelMotor:
    """
    A class to represent a four wheel motor.
    then first two pins are for the left wheel (front and rear wheels respectively) and the last two pins are for the right wheel.
    """

    def __init__(self, speed: int, pins: List[int]):
        GPIO.setmode(GPIO.BCM)
        self.speed = speed
        self.pins = [GPIO.setup(i, GPIO.OUT) for i in pins if i in range(1, 27)]

    def get_speed(self):
        return self.speed

    def get_pins(self):
        return self.pins

    def __str__(self):
        return f"Motor(speed={self.speed}, pins={self.pins})"

    def move_forward(self):
        print("Moving forward")
        GPIO.output([self.pins[0], self.pins[2]], GPIO.LOW)
        GPIO.output([self.pins[1], self.pins[3]], GPIO.HIGH)

    def reverse(self):
        print("Moving backward")
        GPIO.output([self.pins[0], self.pins[2]], GPIO.HIGH)
        GPIO.output([self.pins[1], self.pins[3]], GPIO.LOW)

    def stop(self):
        print("Stopping")
        GPIO.output([self.pins[0], self.pins[2]], GPIO.LOW)
        GPIO.output([self.pins[1], self.pins[3]], GPIO.LOW)

    def turn_left(self):
        print("Turning left")
        GPIO.output([self.pins[0], self.pins[2], self.pins[3]], GPIO.LOW)
        GPIO.output([self.pins[1]], GPIO.LOW)

    def turn_right(self):
        print("Turning right")
        GPIO.output([self.pins[0], self.pins[2], self.pins[1]], GPIO.LOW)
        GPIO.output([self.pins[3]], GPIO.LOW)
