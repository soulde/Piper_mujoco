import numpy as np
from mujoco import MjModel, MjData, mj_step
from renderers import MjRenderer
from controllers import Robot, CartesianIKController, GravityCompensationController
import argparse


def main(args):
    model: MjModel = MjModel.from_xml_path('assets/Piper/scene.xml')
    data: MjData = MjData(model)
    renderer = MjRenderer(model, data, 'human')

    piper = Robot(dof=6, base_link='base_link', end_link='link6', joints=[f'joint{x}' for x in range(1, 7)],
                  actuators=[f'actuator{x}' for x in range(1, 7)])

    if args.mode == 'teach':
        controller = GravityCompensationController(piper)
        target = None
    elif args.mode == 'ik':
        controller = CartesianIKController(piper)
        target = np.array([0.4, 0., 0.20, 0, 0, 1, 0])
    else:
        raise ValueError('mode must be either "teach" or "ik"')

    controller.set_model(model)
    controller.set_data(data)

    while True:
        mj_step(model, data)
        ctrl = controller.step_controller(target)

        def set_control_value(mjdata, ctrl_value, actuators):
            for c, actuator in zip(ctrl_value, actuators):
                mjdata.actuator(actuator).ctrl = c

        set_control_value(mjdata=data, ctrl_value=ctrl, actuators=piper.actuators)
        renderer.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='piper_mujoco_sim',
        description='A simulation for displaying piper with some basic arm controlling',
    )

    parser.add_argument('--mode', default='teach', choices=['teach', 'ik'])
    args = parser.parse_args()
    main(args)
