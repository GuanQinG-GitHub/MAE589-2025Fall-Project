import numpy as np

class ImpedanceScheduler:
    """
    Manages and updates impedance gains (kps and kds) for a robot's joints
    based on its current state, particularly focusing on adjusting ankle gains
    during stance phases.
    """
    def __init__(self, config):
        self.config = config
        self.kp_ankle_pitch_st = config.get('kp_p_st', 40.0)
        self.kp_ankle_roll_st = config.get('kp_r_st', 40.0)
        self.kp_ankle_pitch_sw = config.get('kp_p_sw', 40.0)
        self.kp_ankle_roll_sw = config.get('kp_r_sw', 40.0)
        self.kp_knee_st = config.get('kp_k_st', 40.0)
        self.kp_knee_sw = config.get('kp_k_sw', 40.0)
        
        self.default_kps = np.array(config['kps'], dtype=np.float32).copy()
        self.phase = 0.0


    def update_param(self, new_params):
        """
        Update ankle stiffness parameters.

        Args:
            new_params: List or array with two elements [kp_ankle_pitch, kp_ankle_roll]
        """
        self.kp_ankle_pitch_st = new_params[0]
        self.kp_ankle_roll_st = new_params[1]
        self.kp_ankle_pitch_sw = new_params[2]
        self.kp_ankle_roll_sw = new_params[3]

        
    def update_gains(self, data, model):
        """
        binary switch for ankle stiffness
        """
        kps = self.default_kps.copy()
        # kds are constant in the original code, but we return them for completeness/extensibility
        kds = np.array(self.config['kds'], dtype=np.float32).copy()

        stance_left = is_stance(data, model, side='left')
        stance_right = is_stance(data, model, side='right')

        if stance_left:
            kps[4] = self.kp_ankle_pitch_st
            kps[5] = self.kp_ankle_roll_st
        else:
            kps[4] = self.kp_ankle_pitch_sw
            kps[5] = self.kp_ankle_roll_sw

        if stance_right:
            kps[10] = self.kp_ankle_pitch_st
            kps[11] = self.kp_ankle_roll_st
        else:
            kps[10] = self.kp_ankle_pitch_sw
            kps[11] = self.kp_ankle_roll_sw

        # kd should be changeing with the same ratio as kp
        kds[4] = kds[4] * (kps[4] / self.default_kps[4])
        kds[5] = kds[5] * (kps[5] / self.default_kps[5])
        kds[10] = kds[10] * (kps[10] / self.default_kps[10])
        kds[11] = kds[11] * (kps[11] / self.default_kps[11])
            
        return kps, kds

def is_stance(data, model, side = 'left'):
    # Determine if the requested side leg is in stance (i.e. contacting ground with sufficient force)
    force_threshold = 0.5
    side = side.lower()
    stance = False

    for i in range(data.ncon):
        contact = data.contact[i]

        # MuJoCo contact force slice
        efc_addr = int(contact.efc_address)
        dim = int(contact.dim)
        contact_force_vector = np.array(data.efc_force[efc_addr: efc_addr + dim])

        if np.linalg.norm(contact_force_vector) <= force_threshold:
            continue

        # check if contact involves ground
        g1 = contact.geom1
        g2 = contact.geom2
        name1 = model.geom(g1).name.lower()
        name2 = model.geom(g2).name.lower()

        ground_hit = ("terrain_1" in name1 or "floor" in name1 or
                      "terrain_1" in name2 or "floor" in name2)
        if not ground_hit:
            continue

        # the non-ground geom is the foot
        if "terrain_1" in name1 or "floor" in name1:
            foot_name = name2
        else:
            foot_name = name1

        # detect a foot geom for the requested side (match your XML naming)
        if side in foot_name:
            stance = True
            break

    return bool(stance)

