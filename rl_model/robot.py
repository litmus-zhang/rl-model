from motor import FourWheelMotor as Motor
from sensor import Sensor


def move_robot():
    robot = Motor(0, [14, 15, 18, 23])
    sensor = Sensor(24, 25)

    while True:
        robot.move_forward()

        if sensor.read() < 10:
            robot.reverse()
            robot.turn_left()
            break

    robot.stop()


move_robot()
