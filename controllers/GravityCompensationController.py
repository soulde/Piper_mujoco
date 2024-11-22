import numpy as np
from .BaseController import ArmController


class GravityCompensationController(ArmController):
    def __init__(
            self,
            robot,
    ):
        super().__init__(robot)

        self.name = 'JNTIMP'

        # hyperparameters of impedance controller
        self.B = np.zeros(self.dof)
        self.K = np.zeros(self.dof)

        self.set_jnt_params(
            b=20.0 * np.ones(self.dof),
            k=80.0 * np.ones(self.dof),
        )

    def set_jnt_params(self, b: np.ndarray, k: np.ndarray):
        """ Used for changing the parameters. """
        self.B = b
        self.K = k

    def compute_jnt_torque(
            self,
            q_des: np.ndarray,
            v_des: np.ndarray,
            q_cur: np.ndarray,
            v_cur: np.ndarray,
    ) -> np.ndarray:
        """ robot的关节空间控制的计算公式
            Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_des: desired joint position
        :param v_des: desired joint velocity
        :param q_cur: current joint position
        :param v_cur: current joint velocity
        :return: desired joint torque
        """
        M = self.get_mass_matrix()

        compensation = self.get_coriolis_gravity_compensation()

        # acc_desire = self.K * (q_des - q_cur) + self.B * (v_des - v_cur)
        tau = compensation
        return tau

    def step_controller(self, action=None):
        """

        :param action: joint position
        :return: joint torque
        """

        torque = self.compute_jnt_torque(
            q_des=action,
            v_des=np.zeros(self.dof),
            q_cur=self.get_arm_qpos(),
            v_cur=self.get_arm_qvel(),
        )

        return torque

    def _init_interpolator(self, cfg: dict):
        try:
            from robopal.controllers.planners.interpolators import OTG
        except ImportError:
            raise ImportError("Please install ruckig first: pip install ruckig")
        self.interpolator = OTG(
            OTG_dim=cfg['dof'],
            control_cycle=cfg['control_timestep'],
            max_velocity=0.2,
            max_acceleration=0.4,
            max_jerk=0.6
        )
        self.interpolator.set_params(cfg['init_qpos'], cfg['init_qvel'])

    def step_interpolator(self, action):
        self.interpolator.update_target_position(action)

    def reset(self):
        if self.interpolator is not None:
            self.interpolator.set_params(self.get_arm_qpos(), self.get_arm_qvel())
