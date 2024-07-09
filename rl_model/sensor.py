import Mock.GPIO as GPIO
import time


class Sensor:
    """
    A class to represent a sensor.(ultrasound sensor), which is used to measure distance.
    It has a trigger pin and an echo pin.
    """

    def __init__(self, trigger_pin: int, echo_pin: int):
        GPIO.setmode(GPIO.BCM)
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)

    def read(self) -> float:
        # Ensure trigger pin is low
        GPIO.output(self.trigger_pin, False)
        time.sleep(0.5)

        # Send 10µs pulse to trigger
        GPIO.output(self.trigger_pin, True)
        time.sleep(0.00001)  # 10µs
        GPIO.output(self.trigger_pin, False)

        # Measure the duration of the echo signal
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start

        # Calculate distance
        distance = pulse_duration * 17150  # Speed of sound divided by 2, in cm

        return round(distance, 2)

    def __str__(self):
        return f"Sensor(trigger_pin={self.trigger_pin}, echo_pin={self.echo_pin})"
