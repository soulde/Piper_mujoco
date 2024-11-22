import dataclasses
from copy import deepcopy
from typing import Union, Dict

import mujoco
import numpy as np
from mujoco import MjData
from mujoco import mj_name2id
import controllers.commons.transform as T


class Controller(object):
    def __init__(self):
        self.robot_data: MjData or None = None
        self.robot_model = None
        self.kine_data = None
        self.joint_id2inertialM = []

    def step_controller(self, actions):
        raise NotImplementedError

    def update_data(self, data):
        self.robot_data = data

    def set_model(self, model, joint_id2inertialM=[[0], [1], [2], [3], [4], [5], [6]]):
        self.joint_id2inertialM = joint_id2inertialM
        self.robot_model = model

    def set_data(self, data):
        if self.robot_data is None:
            self.kine_data = deepcopy(data)
            self.update_data(data)


@dataclasses.dataclass
class Robot:
    dof: int
    joints: [str]
    base_link: str
    end_link: str
    actuators: [str]


class ArmController(Controller):
    reference = 'base'

    def __init__(self, robot_type) -> None:
        super().__init__()
        self.name = None
        self.robot_type = robot_type

        # self.robot = robot_type(unique_name=unique_name)
        self.dof = robot_type.dof

        self.init_pos, self.init_quat = None, None
        assert self.reference in ['base', 'world'], "Invalid reference frame."

    def step_controller(
            self, action: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """ Step once computing for all agents.
        """
        raise NotImplementedError

    def reset(self):
        """ reset controller. """
        pass

    def forward_kinematics(self, q):

        def set_joint_qpos(qpos: np.ndarray):
            """ Set joint position. """
            assert qpos.shape[0] == self.dof
            for j, per_arm_joint_names in enumerate([j for j in self.robot_type.joints]):
                self.kine_data.joint(per_arm_joint_names).qpos = qpos[j]

        set_joint_qpos(q)

        mujoco.mj_fwdPosition(self.robot_model, self.kine_data)

        base_pos = self.kine_data.body(self.robot_type.base_link).xpos
        base_mat = self.kine_data.body(self.robot_type.base_link).xmat.reshape(3, 3)
        end_pos = self.kine_data.body(self.robot_type.end_link).xpos
        end_mat = self.kine_data.body(self.robot_type.end_link).xmat.reshape(3, 3)

        end = T.make_transform(end_pos, end_mat)
        base = T.make_transform(base_pos, base_mat)

        if self.reference == 'base':
            ret = np.linalg.pinv(base) @ end
            return ret[:3, -1], T.mat_2_quat(ret[:3, :3])
        else:
            return end_pos, T.mat_2_quat(end_mat)

    def get_arm_qpos(self) -> np.ndarray:
        """ Get arm joint position of the specified agent.

        :param agent: agent name
        :return: joint position
        """
        return np.array(
            [self.robot_data.joint(j).qpos[0] for j in [j for j in self.robot_type.joints]])

    def get_arm_qvel(self) -> np.ndarray:
        """ Get arm joint velocity of the specified agent.

        :param agent: agent name
        :return: joint position
        """
        return np.array(
            [self.robot_data.joint(j).qvel[0] for j in [j for j in self.robot_type.joints]])

    def get_arm_qacc(self) -> np.ndarray:
        """ Get arm joint accelerate of the specified agent.

        :param agent: agent name
        :return: joint position
        """
        return np.array(
            [self.robot_data.joint(j).qacc[0] for j in [j for j in self.robot_type.joints]])

    def get_arm_tau(self, agent: str = 'arm0') -> np.ndarray:
        """ Get arm joint torque of the specified agent.

        :param agent: agent name
        :return: joint torque
        """
        return np.array(
            [self.robot_data.actuator(a).ctrl[0] for a in [j for j in self.robot_type.joints]])

    def get_mass_matrix(self) -> np.ndarray:
        """ Get Mass Matrix
        ref https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/base_controller.py#L61

        :param agent: agent name
        :return: mass matrix
        """
        mass_matrix = np.ndarray(shape=(self.robot_model.nv, self.robot_model.nv), dtype=np.float64, order="C")
        # qM is inertia in joint space
        mujoco.mj_fullM(self.robot_model, mass_matrix, self.robot_data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.robot_data.qvel), len(self.robot_data.qvel)))

        index = sum([self.joint_id2inertialM[i] for i in self.arm_joint_indexes()], [])

        return mass_matrix[index, :][:, index]

    def get_coriolis_gravity_compensation(self) -> np.ndarray:

        index = sum([self.joint_id2inertialM[i] for i in self.arm_joint_indexes()], [])
        return self.robot_data.qfrc_bias[index]

    def arm_joint_indexes(self):

        return [mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in
                [j for j in self.robot_type.joints]]

    def get_end_xpos(self) -> np.ndarray:
        return self.robot_data.body(self.robot_type.end_link).xpos.copy()

    def get_end_xquat(self) -> np.ndarray:
        return self.robot_data.body(self.robot_type.end_link).xquat.copy()

    def get_end_xmat(self) -> np.ndarray:
        return self.robot_data.body(self.robot_type.end_link).xmat.copy().reshape(3, 3)

    def update_init_pose_to_current(self):
        self.init_pos, self.init_quat = self.forward_kinematics(self.get_arm_qpos())

    def get_end_xvel(self) -> np.ndarray:
        """ Computing the end effector velocity

        :param agent: agent name
        :return: end effector velocity, 6*1, [v, w]
        """
        return np.dot(self.get_full_jac(), self.get_arm_qvel())

    def get_base_xpos(self) -> np.ndarray:
        return self.robot_data.body(self.robot_type.base_link).xpos.copy()

    def get_base_xquat(self) -> np.ndarray:
        return self.robot_data.body(self.robot_type.base_link).xquat.copy()

    def get_base_xmat(self) -> np.ndarray:
        return self.robot_data.body(self.robot_type.base_link).xmat.copy().reshape(3, 3)

    def get_full_jac(self) -> np.ndarray:
        """ Computes the full model Jacobian, expressed in the coordinate world frame.

        :param agent: agent name
        :return: Jacobian
        """
        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, self.robot_type.end_link)
        jacp = np.zeros((3, self.robot_model.nv))
        jacr = np.zeros((3, self.robot_model.nv))
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jacp, jacr, bid)
        index = sum([self.joint_id2inertialM[i] for i in self.arm_joint_indexes()], [])

        return np.concatenate([
            jacp[:, index],
            jacr[:, index]
        ], axis=0).copy()

    def get_full_jac_pinv(self) -> np.ndarray:
        """ Computes the full model Jacobian_pinv expressed in the coordinate world frame.

        :param agent: agent name
        :return: Jacobian_pinv
        """
        return np.linalg.pinv(self.get_full_jac()).copy()

    def get_jac_dot(self) -> np.ndarray:
        """ Computing the Jacobian_dot in the joint frame.
        https://github.com/google-deepmind/mujoco/issues/411#issuecomment-1211001685

        :param agent: agent name
        :return: Jacobian_dot
        """
        h = 1e-2
        J = self.get_full_jac()

        original_qpos = self.robot_data.qpos.copy()
        mujoco.mj_integratePos(self.robot_model, self.robot_data.qpos, self.robot_data.qvel, h)
        mujoco.mj_comPos(self.robot_model, self.robot_data)
        mujoco.mj_kinematics(self.robot_model, self.robot_data)

        Jh = self.get_full_jac()
        self.robot_data.qpos = original_qpos

        Jdot = (Jh - J) / h
        return Jdot

    def mani_joint_bounds(self):
        index = self.arm_joint_indexes()
        return (self.robot_model.jnt_range[index, 0],
                self.robot_model.jnt_range[index, 1])
