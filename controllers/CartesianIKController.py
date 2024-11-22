import time

import numpy as np
import mujoco
from mujoco import minimize
from .JointImpedanceController import JointImpedanceController
import controllers.commons.transform as T


class CartesianIKController(JointImpedanceController):

    def __init__(
            self,
            robot,
            is_interpolate=False,
            interpolator_config: dict = None,
            is_pd=False
    ):
        super().__init__(robot, is_interpolate, interpolator_config)

        self.name = 'CARTIK'
        self.reference = 'base'
        self.p_cart = 0.2
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01
        self.is_pd = is_pd
        self.vel_des = np.zeros(3)

    def compute_pd_increment(self, p_goal: np.ndarray,
                             p_cur: np.ndarray,
                             r_goal: np.ndarray,
                             r_cur: np.ndarray,
                             pd_goal: np.ndarray = np.zeros(3),
                             pd_cur: np.ndarray = np.zeros(3)):
        pos_incre = self.p_cart * (p_goal - p_cur) + self.d_cart * (pd_goal - pd_cur)
        quat_incre = self.p_quat * (r_goal - r_cur)
        return pos_incre, quat_incre

    def step_controller(self, action):
        """
        :param: action: end pose
        :return: joint torque
        """

        ret = dict()
        assert len(action) in (3, 7), "Invalid action length."

        p_cur, r_cur = self.forward_kinematics(self.get_arm_qpos())

        if not self.is_pd:
            p_goal = action[:3]
            r_goal = self.init_quat if len(action) == 3 else action[3:]
        else:
            p_cur, r_cur = self.forward_kinematics(self.get_arm_qpos())
            r_target = self.init_quat if len(action) == 3 else action[3:]
            pd_cur = self.get_end_xvel()[:3]
            p_incre, r_incre = self.compute_pd_increment(p_goal=action[:3], p_cur=p_cur,
                                                         r_goal=r_target, r_cur=r_cur,
                                                         pd_goal=self.vel_des, pd_cur=pd_cur[:3])
            p_goal = p_incre + p_cur
            r_goal = r_cur + r_incre
        ret = self.ik(p_goal, r_goal)

        return super().step_controller(ret)

    def ik(self, pos, quat, q_init=None):
        del q_init
        x = self.get_arm_qpos()
        x_prev = x.copy()

        ik_target = lambda x: self._ik_res(x, pos=pos, quat=quat, reg_target=x_prev, radius=.1, reg=1e-2)
        jac_target = lambda x, r: self._ik_jac(x, r, pos=pos, quat=quat, radius=.1, reg=1e-2)

        x, trace = minimize.least_squares(x, ik_target, self.mani_joint_bounds(), jacobian=jac_target, verbose=0)
        return x

    def _ik_res(self, x, pos=None, quat=None, radius=6, reg=1e-3, reg_target=None):
        """Residual for inverse kinematics.

        Args:
            x: joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The residual of the Inverse Kinematics task.
        """
        res = []

        for i in range(x.shape[1]):
            p_cur, r_cur = self.forward_kinematics(x[:, i])

            # Position residual.
            res_pos = p_cur - pos

            # Orientation residual: quaternion difference.
            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, quat, r_cur)

            res_quat *= radius

            # Regularization residual.
            reg_target = self.robot_type.init_qpos if reg_target is None else reg_target
            res_reg = reg * (x[:, i] - reg_target)

            res_i = np.hstack((res_pos, res_quat, res_reg))
            res.append(np.atleast_2d(res_i).T)

        return np.hstack(res)

    def _ik_jac(self, x, res, pos=None, quat=None, radius=.04, reg=1e-3):
        """Analytic Jacobian of inverse kinematics residual

        Args:
            x: joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The Jacobian of the Inverse Kinematics task.
        """
        # least_squares() passes the value of the residual at x which is sometimes
        # useful, but we don't need it here.
        del res
        t1 = time.perf_counter()
        # We can assume x has been copied into qpos
        # and that mj_kinematics has been called by ik()
        # Call mj_comPos (required for Jacobians).
        mujoco.mj_comPos(self.robot_model, self.kine_data)

        # Get end-effector site Jacobian.
        jac_pos = np.empty((3, self.robot_model.nv))
        jac_quat = np.empty((3, self.robot_model.nv))
        mujoco.mj_jacBody(
            self.robot_model, self.kine_data, jac_pos, jac_quat,
            self.kine_data.body(self.robot_type.end_link).id
        )

        index = sum([self.joint_id2inertialM[i] for i in self.arm_joint_indexes()], [])
        jac_pos = jac_pos[:, index]
        jac_quat = jac_quat[:, index]

        if self.reference == 'base':
            base2world_mat = self.get_base_xmat()
            jac_pos = base2world_mat.T @ jac_pos
            jac_quat = base2world_mat.T @ jac_quat

        # Get Deffector, the 3x3 mjd_subQuat Jacobian
        effector_quat = self.kine_data.body(self.robot_type.end_link).xquat

        if self.reference == 'base':
            effector_mat = base2world_mat.T @ T.quat_2_mat(effector_quat)
            effector_quat = T.mat_2_quat(effector_mat)

        Deffector = np.empty((3, 3))
        mujoco.mjd_subQuat(quat, effector_quat, None, Deffector)

        # Rotate into target frame, multiply by subQuat Jacobian, scale by radius.
        target_mat = T.quat_2_mat(quat)
        mat = radius * Deffector.T @ target_mat.T
        jac_quat = mat @ jac_quat

        # Regularization Jacobian.
        jac_reg = reg * np.eye(len(self.arm_joint_indexes()))
        t2 = time.perf_counter()

        return np.vstack((jac_pos, jac_quat, jac_reg))

