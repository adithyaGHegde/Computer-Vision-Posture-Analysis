import turtle
import math
def draw_stick_figure(x, y, angles,val):
    left_elbow,right_elbow,left_knee,right_knee,left_hip,right_hip,left_shoulder,right_shoulder=angles
    arm_length = 50
    leg_length = 75
    left_arm_angle = -135 + left_shoulder
    right_arm_angle = -45 + right_shoulder
    left_leg_angle =  180- (60-left_hip) 
    right_leg_angle = 180- (right_hip+60)
    turtle.penup()
    turtle.goto(x, y + 25)
    turtle.pendown()
    turtle.setheading(270)
    turtle.forward(-25)
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.forward(-25)
    turtle.penup()
    turtle.goto(x, y + 25)
    turtle.pendown()
    turtle.setheading(left_arm_angle)
    turtle.forward(arm_length)
    turtle.penup()
    turtle.goto(x + math.cos(math.radians(left_arm_angle)) * arm_length,
                y + 25 + math.sin(math.radians(left_arm_angle)) * arm_length)
    turtle.pendown()
    turtle.setheading(-135 + left_elbow)
    turtle.forward(25)  
    turtle.penup()
    turtle.goto(x, y + 25)
    turtle.pendown()
    turtle.setheading(right_arm_angle)
    turtle.forward(arm_length)
    turtle.penup()
    turtle.goto(x + math.cos(math.radians(right_arm_angle)) * arm_length,
                y + 25 + math.sin(math.radians(right_arm_angle)) * arm_length)
    turtle.pendown()
    turtle.setheading(-45 + right_elbow)
    turtle.forward(25)
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.setheading(270)
    turtle.forward(50)
    turtle.penup()
    turtle.goto(x, y - 50)
    turtle.pendown()
    turtle.setheading(left_leg_angle)
    turtle.forward(leg_length)
    turtle.penup()
    turtle.goto(x, y - 50)
    turtle.pendown()
    turtle.setheading(right_leg_angle)
    turtle.forward(leg_length)
    turtle.penup()
    turtle.goto(x + math.cos(math.radians(left_leg_angle)) * leg_length,
                y - 50 + math.sin(math.radians(left_leg_angle)) * leg_length)
    turtle.pendown()
    turtle.setheading(180- (90 - left_knee))
    turtle.forward(25)
    
    turtle.penup()
    turtle.goto(x + math.cos(math.radians(right_leg_angle)) * leg_length,
                y - 50 + math.sin(math.radians(right_leg_angle)) * leg_length)
    turtle.pendown()
    turtle.setheading(180 - (90+ right_knee))
    turtle.forward(25)
    



