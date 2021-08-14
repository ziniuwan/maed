import numpy as np

def convert_kps_to_mask(kp_2d, visibility, mask_size):
    """
    Args:
        kp_2d: size (49, 2) range from 0 to 224
        visibility: size (49,)
    """
    img_size = 224

    mask = np.zeros((mask_size,mask_size), dtype=np.float16)
    kp_2d = kp_2d.copy()
    kp_2d = kp_2d // (img_size//mask_size)  # normalize from 0~224 to 0~mask_size
    kp_2d = np.floor(kp_2d).astype(np.int8)
    kp_2d[kp_2d>mask_size-1] = mask_size-1
    kp_2d[kp_2d<0] = 0
    for kp, vis in zip(kp_2d, visibility):
        if vis == 0:
            continue
        mask[kp[1], kp[0]] = 1
    return mask



def keypoint_2d_hflip(kp_2d, img_width):
    """Flip 2d keypoint horizontally around the y-axis
    Args:
        kp_2d: numpy array of shape (T, N, *) or (N, *) where N is the number of keypoints.
        img_width: int
    Return:
        numpy array of the same size of input
    """
    # exchange left limbs and right limbs due to visual chirality
    if len(kp_2d.shape) == 2:
        kp_2d = kp_2d[None,:,:]
    kp_2d = convert_kps(kp_2d, src='spin', dst='spin', flip=True)

    # flip it along y-axis
    kp_2d[:,:,0] = (img_width - 1.) - kp_2d[:,:,0]

    return kp_2d.squeeze()

def keypoint_3d_hflip(kp_3d):
    """Flip 3d keypoint horizontally around the y-axis
    Args:
        kp_3d: (T, N, *) or (N, *) where N is the number of keypoints.
    Return: 
        numpy array of the same size of input
    """
    # exchange left limbs and right limbs due to visual chirality
    if len(kp_3d.shape) == 2:
        kp_3d = kp_3d[None,:,:]
    kp_3d = convert_kps(kp_3d, src='spin', dst='spin', flip=True)

    # flip it along y-axis
    pelvis = (kp_3d[:,27,:] + kp_3d[:, 28,:]) / 2
    kp_3d = kp_3d - pelvis[:, None, :]
    kp_3d[:,:,0] = -kp_3d[:,:,0]
    kp_3d += pelvis[:, None, :]

    return kp_3d.squeeze()

def smpl_pose_hflip(pose):
    """Flip smpl pose parameters
    Args:
        pose: numpy array of shape (T, 72, ) or (72, )
    Return:
        numpy array of the same size of input
    """
    if len(pose.shape) == 1:
        pose = pose[None, :]
    pose_orig = np.reshape(pose,(-1, 24, 3))
    pose_flip = pose_orig.copy()
    for idx in range(pose_flip.shape[1]-1): # skip root joint
        flip_name = get_smpl_joint_names(True)[idx]
        flip_idx = get_smpl_joint_names().index(flip_name)
        
        pose_flip[:, idx, 0] = pose_orig[:, flip_idx, 0]
        pose_flip[:, idx, 1:] = -pose_orig[:, flip_idx, 1:]
    return np.reshape(pose_flip, (-1, 72)).squeeze()



def convert_kps(joints, src, dst, flip=False):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')(flip)

    out_joints = np.zeros((joints.shape[0], len(dst_names), joints.shape[2]))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints[:, idx] = joints[:, src_names.index(jn)]

    return out_joints

def get_perm_idxs(src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()
    idxs = [src_names.index(h) for h in dst_names if h in src_names]
    return idxs

def get_connectivity():
    return [
        ('headtop', 'neck'),
        ('neck', 'lshoulder'),
        ('neck', 'rshoulder'),
        ('rshoulder', 'relbow'),
        ('relbow', 'rwrist'),
        ('lshoulder', 'lelbow'),
        ('lelbow', 'lwrist'),
        ('neck', 'lhip'),
        ('neck', 'rhip'),
        ('rhip', 'rknee'),
        ('rknee', 'rankle'),
        ('lhip', 'lknee'),
        ('lknee', 'lankle')
    ]

def get_mpii3d_test_joint_names():
    return [
        'headtop', # 'head_top',
        'neck',
        'rshoulder',# 'right_shoulder',
        'relbow',# 'right_elbow',
        'rwrist',# 'right_wrist',
        'lshoulder',# 'left_shoulder',
        'lelbow', # 'left_elbow',
        'lwrist', # 'left_wrist',
        'rhip', # 'right_hip',
        'rknee', # 'right_knee',
        'rankle',# 'right_ankle',
        'lhip',# 'left_hip',
        'lknee',# 'left_knee',
        'lankle',# 'left_ankle'
        'hip',# 'pelvis',
        'Spine (H36M)',# 'spine',
        'Head (H36M)',# 'head'
    ]

def get_mpii3d_joint_names():
    return [
        'spine3', # 0,
        'spine4', # 1,
        'spine2', # 2,
        'Spine (H36M)', #'spine', # 3,
        'hip', # 'pelvis', # 4,
        'neck', # 5,
        'Head (H36M)', # 'head', # 6,
        "headtop", # 'head_top', # 7,
        'left_clavicle', # 8,
        "lshoulder", # 'left_shoulder', # 9,
        "lelbow", # 'left_elbow',# 10,
        "lwrist", # 'left_wrist',# 11,
        'left_hand',# 12,
        'right_clavicle',# 13,
        'rshoulder',# 'right_shoulder',# 14,
        'relbow',# 'right_elbow',# 15,
        'rwrist',# 'right_wrist',# 16,
        'right_hand',# 17,
        'lhip', # left_hip',# 18,
        'lknee', # 'left_knee',# 19,
        'lankle', #left ankle # 20
        'left_foot', # 21
        'left_toe', # 22
        "rhip", # 'right_hip',# 23
        "rknee", # 'right_knee',# 24
        "rankle", #'right_ankle', # 25
        'right_foot',# 26
        'right_toe' # 27
    ]

def get_insta_joint_names():
    return [
        'OP RHeel',
        'OP RKnee',
        'OP RHip',
        'OP LHip',
        'OP LKnee',
        'OP LHeel',
        'OP RWrist',
        'OP RElbow',
        'OP RShoulder',
        'OP LShoulder',
        'OP LElbow',
        'OP LWrist',
        'OP Neck',
        'headtop',
        'OP Nose',
        'OP LEye',
        'OP REye',
        'OP LEar',
        'OP REar',
        'OP LBigToe',
        'OP RBigToe',
        'OP LSmallToe',
        'OP RSmallToe',
        'OP LAnkle',
        'OP RAnkle',
    ]

def get_insta_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [4 , 5],
            [6 , 7],
            [7 , 8],
            [8 , 9],
            [9 ,10],
            [2 , 8],
            [3 , 9],
            [10,11],
            [8 ,12],
            [9 ,12],
            [12,13],
            [12,14],
            [14,15],
            [14,16],
            [15,17],
            [16,18],
            [0 ,20],
            [20,22],
            [5 ,19],
            [19,21],
            [5 ,23],
            [0 ,24],
        ])

def get_staf_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [8, 12],
            [12, 13],
            [13, 14],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18],
            [2, 9],
            [5, 12],
            [1, 19],
            [20, 19],
        ]
    )

def get_staf_joint_names():
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]

def get_spin_joint_names(flip=False):
    if flip:
        # spin joints names, flipped version. i.e. all left to right; right to left.
        return [
            'OP Nose',        # 0
            'OP Neck',        # 1
            'OP LShoulder',   # 2
            'OP LElbow',      # 3
            'OP LWrist',      # 4
            'OP RShoulder',   # 5
            'OP RElbow',      # 6
            'OP RWrist',      # 7
            'OP MidHip',      # 8
            'OP LHip',        # 9
            'OP LKnee',       # 10
            'OP LAnkle',      # 11
            'OP RHip',        # 12
            'OP RKnee',       # 13
            'OP RAnkle',      # 14
            'OP LEye',        # 15
            'OP REye',        # 16
            'OP LEar',        # 17
            'OP REar',        # 18
            'OP RBigToe',     # 19
            'OP RSmallToe',   # 20
            'OP RHeel',       # 21
            'OP LBigToe',     # 22
            'OP LSmallToe',   # 23
            'OP LHeel',       # 24
            'lankle',         # 25
            'lknee',          # 26
            'lhip',           # 27
            'rhip',           # 28
            'rknee',          # 29
            'rankle',         # 30
            'lwrist',         # 31
            'lelbow',         # 32
            'lshoulder',      # 33
            'rshoulder',      # 34
            'relbow',         # 35
            'rwrist',         # 36
            'neck',           # 37
            'headtop',        # 38
            'hip',            # 39 'Pelvis (MPII)', # 39
            'thorax',         # 40 'Thorax (MPII)', # 40
            'Spine (H36M)',   # 41
            'Jaw (H36M)',     # 42
            'Head (H36M)',    # 43
            'nose',           # 44
            'reye',           # 45 'Left Eye', # 45
            'leye',           # 46 'Right Eye', # 46
            'rear',           # 47 'Left Ear', # 47
            'lear',           # 48 'Right Ear', # 48
        ]
    else:
        return [
            'OP Nose',        # 0
            'OP Neck',        # 1
            'OP RShoulder',   # 2
            'OP RElbow',      # 3
            'OP RWrist',      # 4
            'OP LShoulder',   # 5
            'OP LElbow',      # 6
            'OP LWrist',      # 7
            'OP MidHip',      # 8
            'OP RHip',        # 9
            'OP RKnee',       # 10
            'OP RAnkle',      # 11
            'OP LHip',        # 12
            'OP LKnee',       # 13
            'OP LAnkle',      # 14
            'OP REye',        # 15
            'OP LEye',        # 16
            'OP REar',        # 17
            'OP LEar',        # 18
            'OP LBigToe',     # 19
            'OP LSmallToe',   # 20
            'OP LHeel',       # 21
            'OP RBigToe',     # 22
            'OP RSmallToe',   # 23
            'OP RHeel',       # 24
            'rankle',         # 25
            'rknee',          # 26
            'rhip',           # 27
            'lhip',           # 28
            'lknee',          # 29
            'lankle',         # 30
            'rwrist',         # 31
            'relbow',         # 32
            'rshoulder',      # 33
            'lshoulder',      # 34
            'lelbow',         # 35
            'lwrist',         # 36
            'neck',           # 37
            'headtop',        # 38
            'hip',            # 39 'Pelvis (MPII)', # 39
            'thorax',         # 40 'Thorax (MPII)', # 40
            'Spine (H36M)',   # 41
            'Jaw (H36M)',     # 42
            'Head (H36M)',    # 43
            'nose',           # 44
            'leye',           # 45 'Left Eye', # 45
            'reye',           # 46 'Right Eye', # 46
            'lear',           # 47 'Left Ear', # 47
            'rear',           # 48 'Right Ear', # 48
        ]

def get_spin2_joint_names():
    return [
        'rankle',         # 0
        'rknee',          # 1
        'rhip',           # 2
        'lhip',           # 3
        'lknee',          # 4
        'lankle',         # 5
        'rwrist',         # 6
        'relbow',         # 7
        'rshoulder',      # 8
        'lshoulder',      # 9
        'lelbow',         # 10
        'lwrist',         # 11
        'neck',           # 12
        'headtop',        # 13
        'hip',            # 14 'Pelvis (MPII)', 
        'thorax',         # 15 'Thorax (MPII)',
        'Spine (H36M)',   # 16
        'Jaw (H36M)',     # 17
        'Head (H36M)',    # 18
        'nose',           # 19
        'leye',           # 20 'Left Eye',
        'reye',           # 21 'Right Eye',
        'lear',           # 22 'Left Ear', 
        'rear',           # 23 'Right Ear',
    ]

def get_h36m_joint_names():
    return [
        'hip',  # 0
        'lhip',  # 1
        'lknee',  # 2
        'lankle',  # 3
        'rhip',  # 4
        'rknee',  # 5
        'rankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]

def get_spin_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
            [0 ,38],
        ]
    )

def get_posetrack_joint_names():
    return [
        "nose",
        "neck",
        "headtop",
        "lear",
        "rear",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhip",
        "rhip",
        "lknee",
        "rknee",
        "lankle",
        "rankle"
    ]

def get_posetrack_original_kp_names():
    return [
        'nose',
        'head_bottom',
        'head_top',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

def get_pennaction_joint_names():
   return [
       "lankle",      # 0
       "lknee",       # 1
       "lhip",        # 2
       "rhip",        # 3
       "rknee",       # 4
       "rankle",      # 5
       "lwrist",      # 6
       "lelbow" ,     # 7
       "lshoulder" ,  # 8
       "rshoulder",   # 9
       "relbow" ,     # 10
       "rwrist",      # 11
       "headtop"      # 12
   ]

"""
def get_pennaction_joint_names():
   return [
       "headtop",   # 0
       "lshoulder", # 1
       "rshoulder", # 2
       "lelbow",    # 3
       "relbow",    # 4
       "lwrist",    # 5
       "rwrist",    # 6
       "lhip" ,     # 7
       "rhip" ,     # 8
       "lknee",     # 9
       "rknee" ,    # 10
       "lankle",    # 11
       "rankle"     # 12
   ]
"""

def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]

def get_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 2 ],
            [ 8, 9 ],
            [ 9, 3 ],
            [ 2, 3 ],
            [ 8, 12],
            [ 9, 10],
            [12, 9 ],
            [10, 11],
            [12, 13],
        ]
    )

def get_coco_joint_names():
    return [
        "nose",      # 0
        "leye",      # 1
        "reye",      # 2
        "lear",      # 3
        "rear",      # 4
        "lshoulder", # 5
        "rshoulder", # 6
        "lelbow",    # 7
        "relbow",    # 8
        "lwrist",    # 9
        "rwrist",    # 10
        "lhip",      # 11
        "rhip",      # 12
        "lknee",     # 13
        "rknee",     # 14
        "lankle",    # 15
        "rankle",    # 16
    ]

def get_coco_skeleton():
    # 0  - nose,
    # 1  - leye,
    # 2  - reye,
    # 3  - lear,
    # 4  - rear,
    # 5  - lshoulder,
    # 6  - rshoulder,
    # 7  - lelbow,
    # 8  - relbow,
    # 9  - lwrist,
    # 10 - rwrist,
    # 11 - lhip,
    # 12 - rhip,
    # 13 - lknee,
    # 14 - rknee,
    # 15 - lankle,
    # 16 - rankle,
    return np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [ 5, 11],
            [ 6, 12],
            [ 5, 6 ],
            [ 5, 7 ],
            [ 6, 8 ],
            [ 7, 9 ],
            [ 8, 10],
            [ 1, 2 ],
            [ 0, 1 ],
            [ 0, 2 ],
            [ 1, 3 ],
            [ 2, 4 ],
            [ 3, 5 ],
            [ 4, 6 ]
        ]
    )

def get_mpii_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "hip",       # 6
        "thorax",    # 7
        "neck",      # 8
        "headtop",   # 9
        "rwrist",    # 10
        "relbow",    # 11
        "rshoulder", # 12
        "lshoulder", # 13
        "lelbow",    # 14
        "lwrist",    # 15
    ]

def get_mpii_skeleton():
    # 0  - rankle,
    # 1  - rknee,
    # 2  - rhip,
    # 3  - lhip,
    # 4  - lknee,
    # 5  - lankle,
    # 6  - hip,
    # 7  - thorax,
    # 8  - neck,
    # 9  - headtop,
    # 10 - rwrist,
    # 11 - relbow,
    # 12 - rshoulder,
    # 13 - lshoulder,
    # 14 - lelbow,
    # 15 - lwrist,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 6 ],
            [ 6, 3 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 9 ],
            [ 7, 12],
            [12, 11],
            [11, 10],
            [ 7, 13],
            [13, 14],
            [14, 15]
        ]
    )

def get_aich_joint_names():
    return [
        "rshoulder", # 0
        "relbow",    # 1
        "rwrist",    # 2
        "lshoulder", # 3
        "lelbow",    # 4
        "lwrist",    # 5
        "rhip",      # 6
        "rknee",     # 7
        "rankle",    # 8
        "lhip",      # 9
        "lknee",     # 10
        "lankle",    # 11
        "headtop",   # 12
        "neck",      # 13
    ]

def get_aich_skeleton():
    # 0  - rshoulder,
    # 1  - relbow,
    # 2  - rwrist,
    # 3  - lshoulder,
    # 4  - lelbow,
    # 5  - lwrist,
    # 6  - rhip,
    # 7  - rknee,
    # 8  - rankle,
    # 9  - lhip,
    # 10 - lknee,
    # 11 - lankle,
    # 12 - headtop,
    # 13 - neck,
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [13, 0 ],
            [13, 3 ],
            [ 0, 6 ],
            [ 3, 9 ]
        ]
    )

def get_3dpw_joint_names():
    return [
        "nose",      # 0
        "thorax",    # 1
        "rshoulder", # 2
        "relbow",    # 3
        "rwrist",    # 4
        "lshoulder", # 5
        "lelbow",    # 6
        "lwrist",    # 7
        "rhip",      # 8
        "rknee",     # 9
        "rankle",    # 10
        "lhip",      # 11
        "lknee",     # 12
        "lankle",    # 13
    ]

def get_3dpw_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 2, 3 ],
            [ 3, 4 ],
            [ 1, 5 ],
            [ 5, 6 ],
            [ 6, 7 ],
            [ 2, 8 ],
            [ 5, 11],
            [ 8, 11],
            [ 8, 9 ],
            [ 9, 10],
            [11, 12],
            [12, 13]
        ]
    )

def get_smplcoco_joint_names():
    return [
        "rankle",    # 0
        "rknee",     # 1
        "rhip",      # 2
        "lhip",      # 3
        "lknee",     # 4
        "lankle",    # 5
        "rwrist",    # 6
        "relbow",    # 7
        "rshoulder", # 8
        "lshoulder", # 9
        "lelbow",    # 10
        "lwrist",    # 11
        "neck",      # 12
        "headtop",   # 13
        "nose",      # 14
        "leye",      # 15
        "reye",      # 16
        "lear",      # 17
        "rear",      # 18
    ]

def get_smplcoco_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 1, 2 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 6, 7 ],
            [ 7, 8 ],
            [ 8, 12],
            [12, 9 ],
            [ 9, 10],
            [10, 11],
            [12, 13],
            [14, 15],
            [15, 17],
            [16, 18],
            [14, 16],
            [ 8, 2 ],
            [ 9, 3 ],
            [ 2, 3 ],
        ]
    )

def get_smpl_joint_names(flip=False):
    if flip:
        return [
            'hips',            # 0
            'rightUpLeg',       # 1
            'leftUpLeg',      # 2
            'spine',           # 3
            'rightLeg',         # 4
            'leftLeg',        # 5
            'spine1',          # 6
            'rightFoot',        # 7
            'leftFoot',       # 8
            'spine2',          # 9
            'rightToeBase',     # 10
            'leftToeBase',    # 11
            'neck',            # 12
            'rightShoulder',    # 13
            'leftShoulder',   # 14
            'head',            # 15
            'rightArm',         # 16
            'leftArm',        # 17
            'rightForeArm',     # 18
            'leftForeArm',    # 19
            'rightHand',        # 20
            'leftHand',       # 21
            'rightHandIndex1',  # 22
            'leftHandIndex1', # 23
        ]
    else:
        return [
            'hips',            # 0
            'leftUpLeg',       # 1
            'rightUpLeg',      # 2
            'spine',           # 3
            'leftLeg',         # 4
            'rightLeg',        # 5
            'spine1',          # 6
            'leftFoot',        # 7
            'rightFoot',       # 8
            'spine2',          # 9
            'leftToeBase',     # 10
            'rightToeBase',    # 11
            'neck',            # 12
            'leftShoulder',    # 13
            'rightShoulder',   # 14
            'head',            # 15
            'leftArm',         # 16
            'rightArm',        # 17
            'leftForeArm',     # 18
            'rightForeArm',    # 19
            'leftHand',        # 20
            'rightHand',       # 21
            'leftHandIndex1',  # 22
            'rightHandIndex1', # 23
        ]

def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )