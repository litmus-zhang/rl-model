from motor import FourWheelMotor as Motor
from sensor import Sensor


class Robot:
    def __init__(self, motor: Motor, sensor: Sensor):
        self.motor = motor
        self.sensor = sensor

    def move_forward(self):
        self.motor.move_forward()

    def reverse(self):
        self.motor.reverse()

    def stop(self):
        self.motor.stop()

    def turn_left(self):
        self.motor.turn_left()

    def turn_right(self):
        self.motor.turn_right()

    def read(self):
        return self.sensor.read()

    def __str__(self):
        return f"Robot(motor={self.motor}, sensor={self.sensor})"


robot = Motor(0, [14, 15, 18, 23])
sensor = Sensor(24, 25)


def move_robot():

    while True:
        robot.move_forward()

        if sensor.read() < 10:
            robot.reverse()
            robot.turn_left()
            break

    robot.stop()


move_robot()
