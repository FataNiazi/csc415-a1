import numpy as np
import quaternion
import skimage
import time
import os
import yaml

# Set EGL rendering for headless mode before importing mujoco
os.environ.setdefault('MUJOCO_GL', 'egl')

from dm_control import mujoco

from domrand.define_flags import FLAGS
from domrand.utils.image import display_image, preproc_image
from domrand.utils.data import get_real_cam_pos
from domrand.utils.modder_dm import TextureModder, CameraModder, LightModder
from domrand.utils.sim import look_at
from domrand.utils.sim import Range, Range3D, rto3d # object type things
from domrand.utils.sim import sample, sample_xyz, sample_joints, sample_light_dir, sample_quat, sample_geom_type, random_quat, jitter_quat, jitter_angle

# GLOSSARY:
# gid = geom_id
# bid = body_id 
class SimManager(object):
    """Object to handle randomization of all relevant properties of Mujoco sim"""
    def __init__(self, filepath, random_params={}, gpu_render=False, gui=False, display_data=False):
        self.model = mujoco.MjModel.from_xml_path(filepath)
        self.data = mujoco.MjData(self.model)
        self.filepath = filepath
        self.gui = gui
        self.display_data = display_data
        # Take the default random params and update anything we need
        self.RANDOM_PARAMS = {}
        self.RANDOM_PARAMS.update(random_params)

        self.viewer = None  # GUI not yet implemented for dm_control
        
        # Create renderer
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Get start state of params to slightly jitter later
        self.START_GEOM_POS = self.model.geom_pos.copy()
        self.START_GEOM_SIZE = self.model.geom_size.copy()
        self.START_GEOM_QUAT = self.model.geom_quat.copy()
        self.START_BODY_POS = self.model.body_pos.copy()
        self.START_BODY_QUAT = self.model.body_quat.copy()
        self.START_MATID = self.model.geom_matid.copy()
        #self.FLOOR_OFFSET = self.model.body_pos[self.model.body_name2id('floor')]

        self.tex_modder = TextureModder(self.model, self.data)
        self.tex_modder.whiten_materials()  # ensures materials won't impact colors
        self.cam_modder = CameraModder(self.model, self.data)
        self.light_modder = LightModder(self.model, self.data)

    def get_data(self):
        self._randomize()
        self._forward()
        gt = self._get_ground_truth()
        cam = self._get_cam_frame(gt) 
        return cam, gt

    def _forward(self):
        """Advances simulator a step (NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK)
        And add some visualization"""
        mujoco.mj_forward(self.model, self.data)

    def _get_ground_truth(self):
        robot_gid = self.model.geom('base_link').id
        obj_gid = self.model.geom('object').id

        obj_pos_in_robot_frame = self.data.geom_xpos[obj_gid] - self.data.geom_xpos[robot_gid]
        return obj_pos_in_robot_frame.astype(np.float32)

    def _get_cam_frame(self, ground_truth=None):
        """Grab an image from the camera (224, 224, 3) to feed into CNN"""
        # Render using the renderer
        self.renderer.update_scene(self.data, camera='camera1')
        cam_img = self.renderer.render()  # Returns RGB image (480, 640, 3)
        
        if self.display_data:
            print(ground_truth)
            display_image(cam_img, mode='preproc')

        cam_img = preproc_image(cam_img)
        return cam_img

    def _randomize(self):
        # Apply ablation flags
        if not FLAGS.no_noise:
            self._rand_textures()
        if not FLAGS.no_camera_rand:
            self._rand_camera()
        self._rand_lights()
        #self._rand_robot()
        self._rand_object()
        self._rand_walls()
        if not FLAGS.no_distractors:
            self._rand_distract()
        # Reset data to apply model changes
        mujoco.mj_resetData(self.model, self.data)

    def _rand_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        bright = np.random.binomial(1, 0.5)
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            self.tex_modder.rand_all(name)
            if bright: 
                self.tex_modder.brighten(name, np.random.randint(0,150))
        # Handle skybox separately if it exists
        try:
            self.tex_modder.rand_all('skybox')
            if bright:
                self.tex_modder.brighten('skybox', np.random.randint(0,150))
        except:
            pass
            
    def _rand_camera(self):
        """Randomize pos, orientation, and fov of camera

        FOVY:
        Kinect2 is 53.8
        ASUS is 45 
        https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/specifications/
        http://smeenk.com/kinect-field-of-view-comparison/
        """
        # Params
        FOVY_R = Range(40, 50)  
        C_R3D = Range3D([-0.05,0.05], [-0.05,0.05], [-0.05,0.05])
        ANG3 = Range3D([-3,3], [-3,3], [-3,3])

        # Look approximately at the robot, but then randomize the orientation around that 
        cam_pos = get_real_cam_pos(FLAGS.real_data_path)
        target_id = self.model.body(FLAGS.look_at).id

        cam_off = 0
        target_off = 0
        quat = look_at(cam_pos+cam_off, self.data.body(FLAGS.look_at).xpos+target_off) 
        quat = jitter_angle(quat, ANG3)

        cam_pos += sample_xyz(C_R3D)

        self.cam_modder.set_quat('camera1', quat)
        self.cam_modder.set_pos('camera1', cam_pos)
        self.cam_modder.set_fovy('camera1', sample(FOVY_R))
    
    def _rand_lights(self):
        """Randomize pos, direction, and lights"""
        X = Range(-1.5, -0.5) 
        Y = Range(-0.6, 0.6)
        Z = Range(1.0, 1.5)
        LIGHT_R3D = Range3D(X, Y, Z)
        LIGHT_UNIF = Range3D(Range(0,1), Range(0,1), Range(0,1))

        for i in range(self.model.nlight):
            name = self.model.light(i).name
            lid = self.model.light(name).id
            # random sample 80% of any given light being on 
            if lid != 0:
                self.light_modder.set_active(name, sample([0,1]) < 0.8)
                self.light_modder.set_dir(name, sample_light_dir())

            self.light_modder.set_pos(name, sample_xyz(LIGHT_R3D))

            spec =    np.array([sample(Range(0.5,1))]*3)
            diffuse = np.array([sample(Range(0.5,1))]*3)
            ambient = np.array([sample(Range(0.5,1))]*3)

            self.light_modder.set_specular(name, spec)
            self.light_modder.set_diffuse(name,  diffuse)
            self.light_modder.set_ambient(name,  ambient)
            self.model.light_castshadow[lid] = sample([0,1]) < 0.5

    def _rand_robot(self):
        """Randomize joint angles and jitter orientation"""
        jnt_shape = self.data.qpos.shape
        self.data.qpos[:] = sample_joints(self.model.jnt_range, jnt_shape)

        robot_gid = self.model.geom('robot_table_link').id
        self.model.geom_quat[robot_gid] = jitter_quat(self.START_GEOM_QUAT[robot_gid], 0.01)

    def _rand_object(self):
        obj_gid = self.model.geom('object').id
        obj_bid = self.model.geom_bodyid[obj_gid]
        table_gid = self.model.geom('object_table').id
        table_bid = self.model.body('object_table').id

        xval = self.model.geom_size[table_gid][0]
        yval = self.model.geom_size[table_gid][1]

        O_X = Range(-xval, xval)
        O_Y = Range(-yval, yval)
        O_Z = Range(0, 0)
        O_R3D = Range3D(O_X, O_Y, O_Z)
        # Modify body position instead of geom position
        self.model.body_pos[obj_bid] = self.START_BODY_POS[obj_bid] + sample_xyz(O_R3D)

    def _rand_walls(self):
        wall_bids = {d: self.model.body('wall_'+d).id for d in 'nesw'}
        window_gid = self.model.geom('west_window').id
       
        WA_X = Range(-0.2, 0.2)
        WA_Y = Range(-0.2, 0.2)
        WA_Z = Range(-0.1, 0.1)
        WA_R3D = Range3D(WA_X, WA_Y, WA_Z)

        WI_X = Range(-0.1, 0.1)
        WI_Y = Range(0, 0)
        WI_Z = Range(-0.5, 0.5)
        WI_R3D = Range3D(WI_X, WI_Y, WI_Z)

        R = Range(0,0)
        P = Range(-10,10)
        Y = Range(0,0)
        RPY_R = Range3D(R,P,Y)

        self.model.geom_quat[window_gid] = sample_quat(RPY_R)
        self.model.geom_pos[window_gid] = self.START_GEOM_POS[window_gid] + sample_xyz(WI_R3D)

        for wall_name in wall_bids:
            gid = wall_bids[wall_name]
            self.model.body_quat[gid] = jitter_quat(self.START_BODY_QUAT[gid], 0.01) 
            self.model.body_pos[gid] = self.START_BODY_POS[gid] + sample_xyz(WA_R3D)


    def _rand_distract(self):
        PREFIX = 'distract'
        geom_names = []
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name.startswith(PREFIX):
                geom_names.append(name)

        # Size range
        SX = Range(0.01, 0.5)
        SY = Range(0.01, 0.9)
        SZ = Range(0.01, 0.5)
        S3D = Range3D(SX, SY, SZ)
        # Back range
        B_PX = Range(-0.5, 2)
        B_PY = Range(-1.5, 2)
        B_PZ = Range(0, 3)
        B_P3D = Range3D(B_PX, B_PY, B_PZ)
        # Front range
        F_PX = Range(-2, -0.5)
        F_PY = Range(-2, 1)
        F_PZ = Range(0, 0.5)
        F_P3D = Range3D(F_PX, F_PY, F_PZ)

        for name in geom_names: 
            gid = self.model.geom(name).id
            pos_range = B_P3D if np.random.binomial(1, 0.5) else F_P3D

            self.model.geom_pos[gid] = sample_xyz(pos_range) 
            self.model.geom_quat[gid] = random_quat() 
            self.model.geom_size[gid] = sample_xyz(S3D, mode='logspace')
            self.model.geom_type[gid] = sample_geom_type()
            self.model.geom_rgba[gid][-1] = np.random.binomial(1, 0.5)


    def _set_visible(self, prefix, range_top, visible):
        """Helper function to set visibility of several objects"""
        if not visible:
            if range_top == 0:
                name = prefix
                gid = self.model.geom(name).id
                self.model.geom_rgba[gid][-1] = 0.0

            for i in range(range_top):
                name = "{}{}".format(prefix, i)
                gid = self.model.geom(name).id
                self.model.geom_rgba[gid][-1] = 0.0
        else:
            if range_top == 0:
                name = prefix
                gid = self.model.geom(name).id
                self.model.geom_rgba[gid][-1] = 1.0

            for i in range(range_top):
                name = "{}{}".format(prefix, i)
                gid = self.model.geom(name).id
                self.model.geom_rgba[gid][-1] = 1.0
