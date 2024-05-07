# demo_6D_pose_yuil_robot

| task    | detail |
| -------- | ------- |
| python code for Yuil robot  |   ubuntu 22, yuil lib  |
| grasping cup  |   grasp a moving cup in the conveyor belt   |

# python code for Yuil robot

Based on [this lib](https://openroboticsalliance.com/en/pc/download), python code can be used for Yuil robot like blew.
```
from yuil_lib import Yuil_robot

real_robot = Yuil_robot()
pos = [0.551, 0.383, 0.277,-3.14, 0.0, 1.56]
real_robot.gripper_close()
real_robot.gripper_open()
real_robot.xyz_move(pos,90)
```
  

# grasping cup

```
python yuil_pvnet_yolo.py
```

# Train a 6D pose estimation model
make sure the 3D model of an object  
![3dmodel](./assets/2.gif)

generate dataset of object   
![datase](./assets/dataset.jpg)

code:
https://github.com/zju3dv/pvnet-rendering