# Piper_mujoco
the mujoco simulation model of agilex robotics Piper arm.

## What this repo have
- A mujoco mjcf file of agilex robotics Piper arm imported from solidworks assembles
- A simulation scene for basic test
- Gravity compensation controller based on mujoco api
- Joint Impedance Controller and Cartesian IK Controller adopted from [robopal](https://github.com/NoneJou072/robopal).

## Installation
```shell
git clone https://github.com/soulde/Piper_mujoco.git
cd Piper_mujoco
pip install -r requirements.txt
```

## Run
```shell
python piper_mujoco_sim.py --mode teach # arm with gravity compensation
or 
python piper_mujoco_sim.py --mode ik # arm with cartesian ik controller
```
## Notice
The controller and renderer are adopted from robopal. This project uses the code pieces in controller part, but only slightly modifies the renderer.
## Reference
[1] Piper_ros https://github.com/agilexrobotics/Piper_ros

[2] robopal https://github.com/NoneJou072/robopal
