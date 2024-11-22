from mujoco import MjData, MjModel, mj_name2id, mj_jacBody, mj_jacSite, mjtObj, mjtState, mj_stateSize, mj_getState, \
    mj_setState, mj_id2name
import numpy as np
from typing import Union, List
import logging
import transform as T

def get_joint_vel(mj_data: MjData, joint_names: list[str]) -> np.ndarray:
    """ Get joint velocity. """
    ret = np.concatenate([np.array(mj_data.joint(j).qvel) for j in joint_names])
    return ret

def set_joint_qpos(mj_data: MjData, joints_names: list[str], qpos: np.ndarray) -> None:
    """ Set joint position. """
    assert qpos.shape[0] == len(joints_names)
    for j, per_arm_joint_names in enumerate(joints_names):
        mj_data.joint(per_arm_joint_names).qpos = qpos[j]
def set_body_pose(mj_data: MjData, body_names: list[str], body_poses: np.ndarray) -> None:
    """ Set body pose. """
    assert body_poses.shape[0] == len(body_names)
    for j, per_arm_body_names in enumerate(body_names):
        mj_data.body(per_arm_body_names).xpos = body_poses[j][:3]
        mj_data.body(per_arm_body_names).xquat = T.mat2quat(body_poses[j][3:])

def set_actuator_ctrl(mj_data: MjData, actuator_names: list[str], torque: np.ndarray) -> None:
    """ Set joint torque. """
    assert torque.shape[0] == len(actuator_names)
    for j, per_arm_actuator_names in enumerate(actuator_names):
        mj_data.actuator(per_arm_actuator_names).ctrl = torque[j]


def set_object_pose(mj_data: MjData, obj_joint_name: str = None, obj_pose: np.ndarray = None) -> None:
    """ Set pose of the object. """
    if isinstance(obj_joint_name, str):
        assert obj_pose.shape[0] == 7
        mj_data.joint(obj_joint_name).qpos = obj_pose


def get_site_id(mj_model: MjModel, name: str) -> int:
    """ Get site id from site name.

    :param mj_model:
    :param name: site name
    :return: site id
    """
    return mj_name2id(mj_model, mjtObj.mjOBJ_SITE, name)


def set_site_pos(mj_model: MjModel, site_name: str = None, site_pos: np.ndarray = None) -> None:
    """ Set pose of the object. """
    if isinstance(site_name, str):
        site_id = get_site_id(mj_model, site_name)
        assert site_pos.shape[0] == 3
        mj_model.site_pos[site_id] = site_pos


def get_body_id(mj_model: MjModel, name: str) -> int:
    """ Get body id from body name.

    :param mj_model:
    :param name: body name
    :return: body id
    """
    return mj_name2id(mj_model, mjtObj.mjOBJ_BODY, name)


def get_body_jacp(mj_model: MjModel, mj_data: MjData, name: str) -> np.ndarray:
    """ Query the position jacobian of a mujoco body using a name string.

    :param mj_model:
    :param mj_data:
    :param name: The name of a mujoco body
    :return: The jacp value of the mujoco body
    """
    bid = get_body_id(mj_model, name)
    jacp = np.zeros((3, mj_model.nv))
    mj_jacBody(mj_model, mj_data, jacp, None, bid)
    return jacp


def get_body_jacr(mj_model: MjModel, mj_data: MjData, name: str) -> np.ndarray:
    """ Query the rotation jacobian of a mujoco body using a name string.

    :param mj_data:
    :param mj_model:
    :param name: The name of a mujoco body
    :return: The jacr value of the mujoco body
    """
    bid = get_body_id(mj_model, name)
    jacr = np.zeros((3, mj_model.nv))
    mj_jacBody(mj_model, mj_data, None, jacr, bid)
    return jacr


def get_body_pos(mj_data: MjData, name: str) -> np.ndarray:
    """ Get body position from body name.

    :param mj_data:
    :param name: body name
    :return: body position
    """
    return mj_data.body(name).xpos.copy()


def get_body_quat(mj_data: MjData, name: str) -> np.ndarray:
    """ Get body quaternion from body name.

    :param mj_data:
    :param name: body name
    :return: body quaternion
    """
    return mj_data.body(name).xquat.copy()


def get_body_rotm(mj_data: MjData, name: str) -> np.ndarray:
    """ Get body rotation matrix from body name.

    :param mj_data:
    :param name: body name
    :return: body rotation matrix
    """
    return mj_data.body(name).xmat.copy().reshape(3, 3)


def get_body_xvelp(mj_model: MjModel, mj_data: MjData, name: str) -> np.ndarray:
    """ Get body velocity from body name.

    :param mj_model:
    :param mj_data:
    :param name: body name
    :return: translational velocity of the body
    """
    jacp = get_body_jacp(mj_model, mj_data, name)
    xvelp = np.dot(jacp, mj_data.qvel)
    return xvelp.copy()


def get_body_xvelr(mj_model: MjModel, mj_data: MjData, name: str) -> np.ndarray:
    """ Get body rotational velocity from body name.

    :param mj_data:
    :param mj_model:
    :param name: body name
    :return: rotational velocity of the body
    """
    jacr = get_body_jacr(mj_model, mj_data, name)
    xvelr = np.dot(jacr, mj_data.qvel)
    return xvelr.copy()


def get_camera_pos(mj_data: MjData, name: str) -> np.ndarray:
    """ Get camera position from camera name.

    :param mj_data:
    :param name: camera name
    :return: camera position
    """
    return mj_data.cam(name).pos.copy()


def get_site_jacp(mj_model: MjModel, mj_data: MjData, name):
    """ Query the position jacobian of a mujoco site using a name string.

    :param mj_model:
    :param mj_data:
    :param name: The name of a mujoco site
    :return: The jacp value of the mujoco site
    """
    sid = get_site_id(mj_model, name)
    jacp = np.zeros((3, mj_model.nv))
    mj_jacSite(mj_model, mj_data, jacp, None, sid)
    return jacp


def get_site_jacr(mj_model: MjModel, mj_data: MjData, name) -> np.ndarray:
    """ Query the rotation jacobian of a mujoco site using a name string.

    :param mj_data:
    :param mj_model:
    :param name: The name of a mujoco site
    :return: The jacr value of the mujoco site
    """
    sid = get_site_id(mj_model, name)
    jacr = np.zeros((3, mj_model.nv))
    mj_jacSite(mj_model, mj_data, None, jacr, sid)
    return jacr


def get_site_pos(mj_data: MjData, name: str) -> np.ndarray:
    """ Get body position from site name.

    :param mj_data:
    :param name: site name
    :return: site position
    """
    return mj_data.site(name).xpos.copy()


def get_site_xvelp(mj_model: MjModel, mj_data: MjData, name: str) -> np.ndarray:
    """ Get site velocity from site name.

    :param mj_data:
    :param mj_model:
    :param name: site name
    :return: translational velocity of the site
    """
    jacp = get_site_jacp(mj_model, mj_data, name)
    xvelp = np.dot(jacp, mj_data.qvel)
    return xvelp.copy()


def get_site_xvelr(mj_model: MjModel, mj_data: MjData, name: str) -> np.ndarray:
    """ Get site rotational velocity from site name.

    :param mj_data:
    :param mj_model:
    :param name: site name
    :return: rotational velocity of the site
    """
    jacr = get_site_jacr(mj_model, mj_data, name)
    xvelr = np.dot(jacr, mj_data.qvel)
    return xvelr.copy()


def get_site_quat(mj_data: MjData, name: str) -> np.ndarray:
    """ Get site quaternion from site name.

    :param mj_data:
    :param name: site name
    :return: site quaternion
    """
    return mj_data.site(name).xquat.copy()


def get_site_rotm(mj_data: MjData, name: str) -> np.ndarray:
    """ Get site rotation matrix from site name.

    :param mj_data:
    :param name: site name
    :return: site rotation matrix
    """
    return mj_data.site(name).xmat.copy().reshape(3, 3)


def get_geom_ids(mj_model: MjModel, name: Union[str, List[str]]) -> Union[int, List[int]]:
    """ Get geometry id from its name.

    :param mj_model:
    :param name: geometry name
    :return: geometry id
    """
    if isinstance(name, str):
        return mj_name2id(mj_model, mjtObj.mjOBJ_GEOM, name)
    else:
        ids = []
        for geom in name:
            id = mj_name2id(mj_model, mjtObj.mjOBJ_GEOM, geom)
            ids.append(id)
        return ids


def save_state(mj_model: MjModel, mj_data: MjData) -> np.ndarray:
    """ Save the state of the mujoco model. """
    spec = mjtState.mjSTATE_INTEGRATION
    size = mj_stateSize(mj_model, spec)
    state = np.empty(size, np.float64)
    mj_getState(mj_model, mj_data, state, spec)
    return state.copy()


def load_state(mj_model: MjModel, mj_data: MjData, state: np.ndarray) -> None:
    """ Load the state of the mujoco model. """
    spec = mjtState.mjSTATE_INTEGRATION
    state = np.array(state.flatten(), np.float64)
    mj_setState(mj_model, mj_data, state, spec)


def is_contact(mj_model: MjModel, mj_data: MjData, geom1: Union[str, List[str]], geom2: Union[str, List[str]],
               verbose=False) -> bool:
    """ Check if two geom or geom list is in contact.

    :param verbose:
    :param mj_data:
    :param mj_model:
    :param geom1: geom name/list
    :param geom2: geom name/list
    :return: True/False
    """
    if isinstance(geom1, str):
        geom1 = [geom1]
    if isinstance(geom2, str):
        geom2 = [geom2]

    if len(mj_data.contact) > 0:
        for i, geom_pair in enumerate(mj_data.contact.geom):
            if geom_pair[0] in geom1 and geom_pair[1] in geom2:
                break
            if geom_pair[0] in geom2 and geom_pair[1] in geom1:
                break
            if verbose:
                contact_info = mj_data.contact[i]
                name1 = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, contact_info.geom1)
                name2 = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, contact_info.geom2)
                dist = contact_info.dist
                logging.info(f"contact geom: {name1} and {name2}")
                logging.info(f"dist: {dist}")
            return True
    return False


def get_relative_transform(mj_data: MjData, body_base: str, body_end: str) -> np.ndarray:
    end_pos = mj_data.body(body_end).xpos
    end_mat = mj_data.body(body_end).xmat.reshape(3, 3)

    base_pos = mj_data.body(body_base).xpos
    base_mat = mj_data.body(body_base).xmat.reshape(3, 3)

    end = T.make_transform(end_pos, end_mat)
    base = T.make_transform(base_pos, base_mat)

    ret = np.linalg.pinv(base) @ end
    return np.concatenate([ret[:3, -1], T.mat_2_quat(ret[:3, :3])])
