import time
from jetbot import Robot
robot = Robot()


def pid(steering_angle):

    while True:
        now = time.time() # current time variable
        dt = now - lastTime
        deviation = steering_angle - 90 # equivalent to angle_to_mid_deg variable
        error = abs(deviation)
        if deviation < 5 and deviation > -5: # do not steer if there is a 10-degree error range
            deviation = 0
            error = 0
            robot.left_motor.value = 0.1 + gain
            robot.right_motor.value = 0.1 + gain

        elif deviation > 5: # steer right if the deviation is positive
            robot.left_motor.value = 0.1 + gain
            robot.right_motor.value = 0 + gain

        elif deviation < -5: # steer left if deviation is negative
            robot.left_motor.value = 0 + gain
            robot.right_motor.value = 0.1 + gain

        derivative = Kd * (error - lastError) / dt
        proportional = Kp * error
        PD = int(0.1 + derivative + proportional)

        spd = abs(PD)
        if spd > 25:
            spd = 25

        lastError = error
        lastTime = time.time()
        print("Gain" + gain)
        print()