# CS159-project
Final project for CS 159 at Caltech.

Some modification to the gym package code has been made: in `gym/envs/classic_control/pendulum.py`, we extended the constructor at line 14-16 to specify max speed and torque :

```
def __init__(self, g=10.0,max_speed=8,max_torque=2.):
  self.max_speed = max_speed
  self.max_torque = max_torque 
```
