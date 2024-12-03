import Adafruit_BBIO.PWM as PWM
import time

motor_pin = "P9_14"  # Replace with your pin

# Start PWM on the motor pin
PWM.start(motor_pin, 50)  # 50% duty cycle (adjust as needed)

# Run the motor for 5 seconds
time.sleep(5)

# Stop PWM
PWM.stop(motor_pin)
PWM.cleanup()  # Clean up the PWM settings
