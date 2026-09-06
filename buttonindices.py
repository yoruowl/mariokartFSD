import time, pygame
pygame.init(); pygame.joystick.init()
j = pygame.joystick.Joystick(0); j.init()
print(j.get_name(), "axes", j.get_numaxes(), "buttons", j.get_numbuttons())
while True:
    pygame.event.pump()
    axes = [round(j.get_axis(i), 2) for i in range(j.get_numaxes())]
    buttons = [j.get_button(i) for i in range(j.get_numbuttons())]
    print("axes", axes, "btns", buttons)
    time.sleep(0.2)