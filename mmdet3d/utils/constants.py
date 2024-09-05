from enum import Enum, unique

from mmdet3d.utils.camera import (
    CylindricalCameraModel,
    CylindricalCameraZoomModel,
    PinholeCameraModel,
    PinholeCameraZoomModel,
    SphericalCameraModel,
)

from enum import Enum, unique

NAME = "nuscenes"
HPARAM_NAME = "NUSCENES"

@unique
class COMMON_CLASS_IDS(int, Enum):
    BACKGROUND = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    TRAILER = 4
    CONSTRUCTION = 5
    PEDESTRIAN = 6
    MOTORCYCLE = 7
    BICYCLE = 8
    TRAFFIC_CONE = 9
    BARRIER = 10
    LARGE_VEHICLE = 11  # Added for compatibility with MSL dataloading
    RIDEABLE_VEHICLE = 12  # Added for compatibility with MSL dataloading
    NOISE = 13
    BACKGROUND_DYNAMIC = 14
    DRIVEABLE_SURFACE = 15
    OTHER_FLAT = 16
    SIDEWALK = 17
    TERRAIN = 18
    MANMADE = 19
    VEGETATION = 20
    RIDER = 21  # Added for compatibility with DST dataloading
    UNKNOWN = -1

@unique
class LABEL_CLASS_KEYS(str, Enum):
    # taken from https://www.nuscenes.org/nuscenes
    ANIMAL = "animal"
    PED_ADULT = "human.pedestrian.adult"
    PED_CHILD = "human.pedestrian.child"
    PED_CONSTRUCTION_WORKER = "human.pedestrian.construction_worker"
    PED_PERSONAL_MOBILITY = "human.pedestrian.personal_mobility"
    PED_POLICE_OFFICER = "human.pedestrian.police_officer"
    PED_STROLLER = "human.pedestrian.stroller"
    PED_WHEELCHAIR = "human.pedestrian.wheelchair"
    MO_BARRIER = "movable_object.barrier"
    MO_DEBRIS = "movable_object.debris"
    MO_PUSHPULLABLE = "movable_object.pushable_pullable"
    MO_TRAFFIC_CONE = "movable_object.trafficcone"
    SO_BICYCLE_RACK = "static_object.bicycle_rack"
    BICYCLE = "vehicle.bicycle"
    BUS_BENDY = "vehicle.bus.bendy"
    BUS_RIGID = "vehicle.bus.rigid"
    CAR = "vehicle.car"
    CONSTRUCTION = "vehicle.construction"
    EGO = "vehicle.ego"
    AMBULANCE = "vehicle.emergency.ambulance"
    POLICE = "vehicle.emergency.police"
    STATIC_OTHER = "static.other"
    MOTORCYCLE = "vehicle.motorcycle"
    TRAILER = "vehicle.trailer"
    TRUCK = "vehicle.truck"
    # Lidar Seg extras
    NOISE = "noise"
    DRIVEABLE_SURFACE = "flat.driveable_surface"
    OTHER_FLAT = "flat.other"
    SIDEWALK = "flat.sidewalk"
    TERRAIN = "flat.terrain"
    MANMADE = "static.manmade"
    VEGETATION = "static.vegetation"


CLASS_LABEL_MAPPING = {
    # Class names available from the dataset: Common class id
    LABEL_CLASS_KEYS.ANIMAL: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.PED_ADULT: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_CHILD: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_CONSTRUCTION_WORKER: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_PERSONAL_MOBILITY: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.PED_POLICE_OFFICER: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_STROLLER: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    LABEL_CLASS_KEYS.PED_WHEELCHAIR: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    LABEL_CLASS_KEYS.MO_BARRIER: COMMON_CLASS_IDS.BARRIER,
    LABEL_CLASS_KEYS.MO_DEBRIS: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.MO_PUSHPULLABLE: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.MO_TRAFFIC_CONE: COMMON_CLASS_IDS.TRAFFIC_CONE,
    LABEL_CLASS_KEYS.SO_BICYCLE_RACK: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.BICYCLE: COMMON_CLASS_IDS.BICYCLE,
    LABEL_CLASS_KEYS.BUS_BENDY: COMMON_CLASS_IDS.BUS,
    LABEL_CLASS_KEYS.BUS_RIGID: COMMON_CLASS_IDS.BUS,
    LABEL_CLASS_KEYS.CAR: COMMON_CLASS_IDS.CAR,
    LABEL_CLASS_KEYS.CONSTRUCTION: COMMON_CLASS_IDS.CONSTRUCTION,
    LABEL_CLASS_KEYS.EGO: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.AMBULANCE: COMMON_CLASS_IDS.CAR,
    LABEL_CLASS_KEYS.POLICE: COMMON_CLASS_IDS.CAR,
    LABEL_CLASS_KEYS.STATIC_OTHER: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.MOTORCYCLE: COMMON_CLASS_IDS.MOTORCYCLE,
    LABEL_CLASS_KEYS.TRAILER: COMMON_CLASS_IDS.TRAILER,
    LABEL_CLASS_KEYS.TRUCK: COMMON_CLASS_IDS.TRUCK,
}

LIDAR_LABEL_MAPPING = {
    LABEL_CLASS_KEYS.ANIMAL: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.PED_ADULT: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_CHILD: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_CONSTRUCTION_WORKER: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_PERSONAL_MOBILITY: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.PED_POLICE_OFFICER: COMMON_CLASS_IDS.PEDESTRIAN,
    LABEL_CLASS_KEYS.PED_STROLLER: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.PED_WHEELCHAIR: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.MO_BARRIER: COMMON_CLASS_IDS.BARRIER,
    LABEL_CLASS_KEYS.MO_DEBRIS: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.MO_PUSHPULLABLE: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.MO_TRAFFIC_CONE: COMMON_CLASS_IDS.TRAFFIC_CONE,
    LABEL_CLASS_KEYS.SO_BICYCLE_RACK: COMMON_CLASS_IDS.BACKGROUND_DYNAMIC,
    LABEL_CLASS_KEYS.BICYCLE: COMMON_CLASS_IDS.BICYCLE,
    LABEL_CLASS_KEYS.BUS_BENDY: COMMON_CLASS_IDS.BUS,
    LABEL_CLASS_KEYS.BUS_RIGID: COMMON_CLASS_IDS.BUS,
    LABEL_CLASS_KEYS.CAR: COMMON_CLASS_IDS.CAR,
    LABEL_CLASS_KEYS.CONSTRUCTION: COMMON_CLASS_IDS.CONSTRUCTION,
    LABEL_CLASS_KEYS.EGO: COMMON_CLASS_IDS.NOISE,
    LABEL_CLASS_KEYS.AMBULANCE: COMMON_CLASS_IDS.CAR,
    LABEL_CLASS_KEYS.POLICE: COMMON_CLASS_IDS.CAR,
    LABEL_CLASS_KEYS.STATIC_OTHER: COMMON_CLASS_IDS.BACKGROUND,
    LABEL_CLASS_KEYS.MOTORCYCLE: COMMON_CLASS_IDS.MOTORCYCLE,
    LABEL_CLASS_KEYS.TRAILER: COMMON_CLASS_IDS.TRAILER,
    LABEL_CLASS_KEYS.TRUCK: COMMON_CLASS_IDS.TRUCK,
    LABEL_CLASS_KEYS.NOISE: COMMON_CLASS_IDS.NOISE,
    LABEL_CLASS_KEYS.DRIVEABLE_SURFACE: COMMON_CLASS_IDS.DRIVEABLE_SURFACE,
    LABEL_CLASS_KEYS.OTHER_FLAT: COMMON_CLASS_IDS.OTHER_FLAT,
    LABEL_CLASS_KEYS.SIDEWALK: COMMON_CLASS_IDS.SIDEWALK,
    LABEL_CLASS_KEYS.TERRAIN: COMMON_CLASS_IDS.TERRAIN,
    LABEL_CLASS_KEYS.MANMADE: COMMON_CLASS_IDS.MANMADE,
    LABEL_CLASS_KEYS.VEGETATION: COMMON_CLASS_IDS.VEGETATION,
}


DIM_INDEX_IN_CONFIG = {
    "CAM_FRONT_LEFT": 0,
    "CAM_FRONT": 0,
    "CAM_FRONT_RIGHT": 0,
    "CAM_BACK_LEFT": 0,
    "CAM_BACK": 1,
    "CAM_BACK_RIGHT": 0,
}


@unique
class DATA_DEFINITION_KEYS(str, Enum):
    VERSION = "dataset_version"
    NAME = "dataset_name"
    ROOT = "dataset_root"
    OCCUPANCY_ROOT = "occupancy_root"
    IS_TRAINING = "is_training"

@unique
class VISIBILITY_MAPPING(int, Enum):
    VISIBILITY_0_40 = 1
    VISIBILITY_40_60 = 2
    VISIBILITY_60_80 = 3
    VISIBILITY_80_100 = 4


OCCLUSION_MAPPING = {
    1: VISIBILITY_MAPPING.VISIBILITY_0_40,
    2: VISIBILITY_MAPPING.VISIBILITY_40_60,
    3: VISIBILITY_MAPPING.VISIBILITY_60_80,
    4: VISIBILITY_MAPPING.VISIBILITY_80_100,
}


VISIBILITY_LOWER_BOUND = {
    None: 0.0,  # no visibility label is mapped to 0.0
    1: 0.0,  # VISIBILITY_MAPPING.VISIBILITY_0_40
    2: 0.4,  # VISIBILITY_MAPPING.VISIBILITY_40_60,
    3: 0.6,  # VISIBILITY_MAPPING.VISIBILITY_60_80,
    4: 0.8,  # VISIBILITY_MAPPING.VISIBILITY_80_100,
}


@unique
class ANNOTATION_KEYS(str, Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    SIZE = "size"
    CATEGORY_NAME = "category_name"
    INSTANCE_TOKEN = "instance_token"
    VISIBILITY_TOKEN = "visibility_token"


@unique
class DATASET_CONFIG(str, Enum):
    TRAIN = "train"
    VAL = "val"
    MINI_VERSION = "mini"


@unique
class SAMPLE_KEYS(str, Enum):
    TOKEN = "token"
    SCENE_TOKEN = "scene_token"
    SAMPLE_TOKEN = "sample_token"
    TIMESTAMP = "timestamp"
    SAMPLE_ANNOTATION = "sample_annotation"
    ANNOTATION_LIST = "anns"
    NAME = "name"
    DATA = "data"
    FILENAME = "filename"


@unique
class SCENE_KEYS(str, Enum):
    ROW_NAME = "name"
    SAMPLE = "sample"
    SAMPLE_DATA = "sample_data"
    LIDAR_TOP = "LIDAR_TOP"
    EGO_POSE = "ego_pose"
    EGO_POSE_TOKEN = "ego_pose_token"
    CALIBRATED_SENSOR = "calibrated_sensor"
    CALIBRATED_SENSOR_TOKEN = "calibrated_sensor_token"
    CAMERA_INTRINSIC = "camera_intrinsic"
    SCENE = "scene"


CLASSES_WITHOUT_ORIENTATION = [
    CLASS_LABEL_MAPPING[LABEL_CLASS_KEYS.MO_TRAFFIC_CONE],
]

CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

DEFAULT_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZATION_STD = [0.229, 0.224, 0.225]

SEGMENTATION_2D_BIAS_INIT = [2.0, -2.0, 1.0]  # per class bias init (bg, objects, road surface)


@unique
class BACKBONE_NET_VERSION(str, Enum):
    effnet_b4 = "efficientnet-b4"
    effnet_b0 = "efficientnet-b0"
    resnet50 = "resnet50"
    vit_small_patch14_dinov2 = "vit_small_patch14_dinov2"
    vit_base_patch14_dinov2 = "vit_base_patch14_dinov2"
    vit_base_patch14_reg4_dinov2 = "vit_base_patch14_reg4_dinov2"
    vit_large_patch14_dinov2 = "vit_large_patch14_dinov2"
    eva02_base_patch16_clip_224 = "eva02_base_patch16_clip_224"
    eva02_large_patch14_clip_336 = "eva02_large_patch14_clip_336"
    vit_base_patch16_384 = "vit_base_patch16_384"
    timm_effnet_v2_s = "timm_tf_efficientnetv2_s"
    timm_resnet50 = "timm_resnet50"
    timm_convnext_tiny = "timm_convnext_tiny"
    timm_convnext_small = "timm_convnext_small"
    timm_convnext_base = "timm_convnext_base"
    mmdet_intern_image_s = "mmdet_intern_image_s"
    mmdet_intern_image_b = "mmdet_intern_image_b"
    mmdet_vovnet = "mmdet_vovnet"


@unique
class BACKBONE_NECK(str, Enum):
    bifpn = "bifpn"
    secondfpn = "secondfpn"


@unique
class TRANSFORMATION_KEYS(str, Enum):
    NONE = "none"
    LIFT_SPLAT_SHOOT = "lift_splat_shoot"
    FULLY_CONNECTED = "fully_connected"
    TRANSFORMER = "transformer"
    TRANSFORMER_COLUMN = "transformer_column"
    TRANSFORMER_CAMERA = "transformer_camera"
    TRANSFORMER_FULL = "transformer_full"
    CONTEXT = "context"
    SHIFT_SHAFT_SHEESH = "shift_shaft_sheesh"
    SHIFT_SHAFT_SHEESH_3D = "shift_shaft_sheesh_3d"
    SPLAT_3D = "splat_3d"
    BEV_TO_3D = "bev_to_3d"


@unique
class POSITIONAL_ENCODING_KEYS(str, Enum):
    NONE = "none"
    CONCAT = "concat"
    ADD_1D = "add_1D"
    ADD_ND = "add_ND"
    ADD_3D_OLD = "add_3D_old"
    POLAR = "polar"


@unique
class FILTER_ACTIONS(str, Enum):
    DISCARD = "discard"
    IGNORE = "ignore"


MODEL_NO_DEPTH_REQUIRED = (
    TRANSFORMATION_KEYS.FULLY_CONNECTED,
    TRANSFORMATION_KEYS.TRANSFORMER,
    TRANSFORMATION_KEYS.TRANSFORMER_COLUMN,
    TRANSFORMATION_KEYS.TRANSFORMER_CAMERA,
    TRANSFORMATION_KEYS.TRANSFORMER_FULL,
    TRANSFORMATION_KEYS.CONTEXT,
)


@unique
class BEV_DIM_INDICES(int, Enum):
    LENGTH = 0
    WIDTH = 1


@unique
class DETECTOR_HEAD_TYPE(str, Enum):
    SIMPLE_BOX_3D_HEAD = "simple_box_3D_head"
    BEVDEPTH_HEAD = "bevdepth_head"
    CENTER_POINT_HEAD_2D = "center_point_head_2d"




# Used only during evaluation and visualisation
COMMON_CLASS_KEYS = {
    COMMON_CLASS_IDS.BACKGROUND: "background",
    COMMON_CLASS_IDS.CAR: "Car",
    COMMON_CLASS_IDS.TRUCK: "Truck",
    COMMON_CLASS_IDS.BUS: "Bus",
    COMMON_CLASS_IDS.TRAILER: "Trailer",
    COMMON_CLASS_IDS.CONSTRUCTION: "Construction",
    COMMON_CLASS_IDS.PEDESTRIAN: "Pedestrian",
    COMMON_CLASS_IDS.MOTORCYCLE: "Motorcycle",
    COMMON_CLASS_IDS.BICYCLE: "Bicycle",
    COMMON_CLASS_IDS.TRAFFIC_CONE: "Traffic_cone",
    COMMON_CLASS_IDS.BARRIER: "Barrier",
    COMMON_CLASS_IDS.LARGE_VEHICLE: "Truck",  # Needs to be mappable to a key known by the LH5 nuScenes evaluation.
    COMMON_CLASS_IDS.RIDEABLE_VEHICLE: "Bicycle",  # Needs to be mappable to a key known by the LH5 nuScenes evaluation.
    COMMON_CLASS_IDS.RIDER: "Bicycle",  # Needs to be mappable to a key known by the LH5 nuScenes evaluation.
}


CLASS_LABEL_MAPPING_NUSCENES = {
    COMMON_CLASS_IDS.BACKGROUND: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.CAR: COMMON_CLASS_IDS.CAR,
    COMMON_CLASS_IDS.TRUCK: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.BUS: COMMON_CLASS_IDS.BUS,
    COMMON_CLASS_IDS.TRAILER: COMMON_CLASS_IDS.TRAILER,
    COMMON_CLASS_IDS.CONSTRUCTION: COMMON_CLASS_IDS.CONSTRUCTION,
    COMMON_CLASS_IDS.PEDESTRIAN: COMMON_CLASS_IDS.PEDESTRIAN,
    COMMON_CLASS_IDS.MOTORCYCLE: COMMON_CLASS_IDS.MOTORCYCLE,
    COMMON_CLASS_IDS.BICYCLE: COMMON_CLASS_IDS.BICYCLE,
    COMMON_CLASS_IDS.TRAFFIC_CONE: COMMON_CLASS_IDS.TRAFFIC_CONE,
    COMMON_CLASS_IDS.BARRIER: COMMON_CLASS_IDS.BARRIER,
    COMMON_CLASS_IDS.LARGE_VEHICLE: COMMON_CLASS_IDS.LARGE_VEHICLE,
    COMMON_CLASS_IDS.RIDEABLE_VEHICLE: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.RIDER: COMMON_CLASS_IDS.PEDESTRIAN,
}


CLASS_LABEL_MAPPING_L4 = {
    COMMON_CLASS_IDS.BACKGROUND: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.CAR: COMMON_CLASS_IDS.CAR,
    COMMON_CLASS_IDS.TRUCK: COMMON_CLASS_IDS.LARGE_VEHICLE,
    COMMON_CLASS_IDS.BUS: COMMON_CLASS_IDS.LARGE_VEHICLE,
    COMMON_CLASS_IDS.TRAILER: COMMON_CLASS_IDS.LARGE_VEHICLE,
    COMMON_CLASS_IDS.CONSTRUCTION: COMMON_CLASS_IDS.LARGE_VEHICLE,
    COMMON_CLASS_IDS.PEDESTRIAN: COMMON_CLASS_IDS.PEDESTRIAN,
    COMMON_CLASS_IDS.MOTORCYCLE: COMMON_CLASS_IDS.MOTORCYCLE,
    COMMON_CLASS_IDS.BICYCLE: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    COMMON_CLASS_IDS.TRAFFIC_CONE: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.BARRIER: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.LARGE_VEHICLE: COMMON_CLASS_IDS.LARGE_VEHICLE,
    COMMON_CLASS_IDS.RIDEABLE_VEHICLE: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    COMMON_CLASS_IDS.RIDER: COMMON_CLASS_IDS.PEDESTRIAN,
}


CLASS_LABEL_MAPPING_DST = {
    COMMON_CLASS_IDS.BACKGROUND: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.CAR: COMMON_CLASS_IDS.CAR,
    COMMON_CLASS_IDS.TRUCK: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.BUS: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.TRAILER: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.CONSTRUCTION: COMMON_CLASS_IDS.CONSTRUCTION,  # construction will be ignored
    COMMON_CLASS_IDS.PEDESTRIAN: COMMON_CLASS_IDS.PEDESTRIAN,
    COMMON_CLASS_IDS.MOTORCYCLE: COMMON_CLASS_IDS.MOTORCYCLE,  # motorcycle will be ignored
    COMMON_CLASS_IDS.BICYCLE: COMMON_CLASS_IDS.BICYCLE,  # bicycle will be ignored
    COMMON_CLASS_IDS.TRAFFIC_CONE: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.BARRIER: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.LARGE_VEHICLE: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.RIDEABLE_VEHICLE: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,  # rideable vehicle will be ignored
    COMMON_CLASS_IDS.RIDER: COMMON_CLASS_IDS.RIDER,
}


CLASS_LABEL_MAPPING_KSS = {
    COMMON_CLASS_IDS.BACKGROUND: COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.CAR: COMMON_CLASS_IDS.CAR,
    COMMON_CLASS_IDS.TRUCK: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.BUS: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.TRAILER: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.CONSTRUCTION: COMMON_CLASS_IDS.CONSTRUCTION,  # construction will be ignored, decision pending
    COMMON_CLASS_IDS.PEDESTRIAN: COMMON_CLASS_IDS.PEDESTRIAN,
    COMMON_CLASS_IDS.MOTORCYCLE: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    COMMON_CLASS_IDS.BICYCLE: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    COMMON_CLASS_IDS.TRAFFIC_CONE: COMMON_CLASS_IDS.TRAFFIC_CONE,  # traffic cone will be ignored
    COMMON_CLASS_IDS.BARRIER: COMMON_CLASS_IDS.BARRIER,  # barrier will be ignored
    COMMON_CLASS_IDS.LARGE_VEHICLE: COMMON_CLASS_IDS.TRUCK,
    COMMON_CLASS_IDS.RIDEABLE_VEHICLE: COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    COMMON_CLASS_IDS.RIDER: COMMON_CLASS_IDS.RIDER,
}


@unique
class CLASS_SELECTION_CONFIG_KEYS(str, Enum):
    ALL = "all"
    BG_VS_DYNAMIC = "bg_vs_dynamic"
    VEHICLES = "vehicles"
    CARS = "cars"
    L4 = "l4"
    CENTERPOINT_NUSCENES = "centerpoint_nuscenes"
    CENTERPOINT_LH5 = "centerpoint_lh5"
    CENTERPOINT_DST = "centerpoint_dst"
    CENTERPOINT_KSS = "centerpoint_kss"


# Classes in CLASS_SELECTION_CONFIG_KEYS.BG_VS_DYNAMIC (see below)
# will be mapped to dynamic here.
@unique
class BG_VS_DYNAMIC_CLASS_IDS(int, Enum):
    BACKGROUND = 0
    DYNAMIC = 1


# Defines relevant classes for the specific configuration
CLASS_SELECTION_CONFIG = {
    CLASS_SELECTION_CONFIG_KEYS.ALL: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.BUS,
        COMMON_CLASS_IDS.TRAILER,
        COMMON_CLASS_IDS.CONSTRUCTION,
        COMMON_CLASS_IDS.PEDESTRIAN,
        COMMON_CLASS_IDS.MOTORCYCLE,
        COMMON_CLASS_IDS.BICYCLE,
        COMMON_CLASS_IDS.TRAFFIC_CONE,
        COMMON_CLASS_IDS.BARRIER,
        COMMON_CLASS_IDS.LARGE_VEHICLE,
        COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    ],
    CLASS_SELECTION_CONFIG_KEYS.BG_VS_DYNAMIC: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.BUS,
        COMMON_CLASS_IDS.TRAILER,
        COMMON_CLASS_IDS.CONSTRUCTION,
        COMMON_CLASS_IDS.MOTORCYCLE,
        COMMON_CLASS_IDS.BICYCLE,
    ],
    CLASS_SELECTION_CONFIG_KEYS.VEHICLES: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.BUS,
        COMMON_CLASS_IDS.TRAILER,
        COMMON_CLASS_IDS.CONSTRUCTION,
        COMMON_CLASS_IDS.MOTORCYCLE,
        COMMON_CLASS_IDS.BICYCLE,
    ],
    CLASS_SELECTION_CONFIG_KEYS.CARS: [
        COMMON_CLASS_IDS.CAR,
    ],
    CLASS_SELECTION_CONFIG_KEYS.L4: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.LARGE_VEHICLE,
        COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
        COMMON_CLASS_IDS.MOTORCYCLE,
    ],
    CLASS_SELECTION_CONFIG_KEYS.CENTERPOINT_NUSCENES: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.CONSTRUCTION,
        COMMON_CLASS_IDS.BUS,
        COMMON_CLASS_IDS.TRAILER,
        COMMON_CLASS_IDS.BARRIER,
        COMMON_CLASS_IDS.MOTORCYCLE,
        COMMON_CLASS_IDS.BICYCLE,
        COMMON_CLASS_IDS.PEDESTRIAN,
        COMMON_CLASS_IDS.TRAFFIC_CONE,
    ],
    CLASS_SELECTION_CONFIG_KEYS.CENTERPOINT_LH5: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.CONSTRUCTION,
        COMMON_CLASS_IDS.BUS,
        COMMON_CLASS_IDS.TRAILER,
        COMMON_CLASS_IDS.MOTORCYCLE,
        COMMON_CLASS_IDS.BICYCLE,
        COMMON_CLASS_IDS.PEDESTRIAN,
    ],
    CLASS_SELECTION_CONFIG_KEYS.CENTERPOINT_DST: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.PEDESTRIAN,
        COMMON_CLASS_IDS.RIDER,
    ],
    CLASS_SELECTION_CONFIG_KEYS.CENTERPOINT_KSS: [
        COMMON_CLASS_IDS.CAR,
        COMMON_CLASS_IDS.TRUCK,
        COMMON_CLASS_IDS.PEDESTRIAN,
        COMMON_CLASS_IDS.RIDEABLE_VEHICLE,
    ],
}

STATIC_CLASSES = [
    COMMON_CLASS_IDS.BACKGROUND,
    COMMON_CLASS_IDS.DRIVEABLE_SURFACE,
    COMMON_CLASS_IDS.OTHER_FLAT,
    COMMON_CLASS_IDS.SIDEWALK,
    COMMON_CLASS_IDS.TERRAIN,
    COMMON_CLASS_IDS.MANMADE,
    COMMON_CLASS_IDS.VEGETATION,
]

OCCUPANCY_CLASS_SELECTION = [
    [COMMON_CLASS_IDS.BACKGROUND, COMMON_CLASS_IDS.UNKNOWN],
    [COMMON_CLASS_IDS.CAR],
    [COMMON_CLASS_IDS.TRUCK],
    [COMMON_CLASS_IDS.BUS],
    [COMMON_CLASS_IDS.TRAILER],
    [COMMON_CLASS_IDS.CONSTRUCTION],
    [COMMON_CLASS_IDS.PEDESTRIAN],
    [COMMON_CLASS_IDS.MOTORCYCLE],
    [COMMON_CLASS_IDS.BICYCLE],
    [COMMON_CLASS_IDS.TRAFFIC_CONE],
    [COMMON_CLASS_IDS.BARRIER],
    [COMMON_CLASS_IDS.DRIVEABLE_SURFACE],
    [COMMON_CLASS_IDS.OTHER_FLAT],
    [COMMON_CLASS_IDS.SIDEWALK],
    [COMMON_CLASS_IDS.TERRAIN],
    [COMMON_CLASS_IDS.MANMADE],
    [COMMON_CLASS_IDS.VEGETATION],
]

OCCUPANCY_CLASS_SELECTION_BOXES = [
    [COMMON_CLASS_IDS.BACKGROUND],
    [COMMON_CLASS_IDS.CAR],
    [COMMON_CLASS_IDS.TRUCK],
    [COMMON_CLASS_IDS.BUS],
    [COMMON_CLASS_IDS.TRAILER],
    [COMMON_CLASS_IDS.CONSTRUCTION],
    [COMMON_CLASS_IDS.PEDESTRIAN],
    [COMMON_CLASS_IDS.MOTORCYCLE],
    [COMMON_CLASS_IDS.BICYCLE],
    [COMMON_CLASS_IDS.TRAFFIC_CONE],
    [COMMON_CLASS_IDS.BARRIER],
]


DEFAULT_ATTRIBUTES = {
    **{k: "" for k in COMMON_CLASS_IDS},
    COMMON_CLASS_IDS.CAR: "vehicle.parked",
    COMMON_CLASS_IDS.PEDESTRIAN: "pedestrian.moving",
    COMMON_CLASS_IDS.TRAILER: "vehicle.parked",
    COMMON_CLASS_IDS.TRUCK: "vehicle.parked",
    COMMON_CLASS_IDS.BUS: "vehicle.moving",
    COMMON_CLASS_IDS.MOTORCYCLE: "cycle.without_rider",
    COMMON_CLASS_IDS.CONSTRUCTION: "vehicle.parked",
    COMMON_CLASS_IDS.BICYCLE: "cycle.without_rider",
    COMMON_CLASS_IDS.BARRIER: "",
    COMMON_CLASS_IDS.TRAFFIC_CONE: "",
    COMMON_CLASS_IDS.LARGE_VEHICLE: "vehicle.parked",
    COMMON_CLASS_IDS.RIDEABLE_VEHICLE: "cycle.without_rider",
}


@unique
class CROP_INDICES(int, Enum):
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3


@unique
class GT_DATA_KEYS(str, Enum):
    SCENE_NAME = "scene_name"
    SAMPLE_TOKEN = "sample_token"
    IMAGE = "image"
    CAMERA_MODEL = "camera_model"
    POINT_CLOUD = "point_cloud"
    POINT_CLOUD_ORIGIN = "point_cloud_origin"
    RANGE_VIEW = "range_view"
    OCCUPANCY = "occupancy"
    BEV_SURFACE = "bev_surface"
    INTRINSICS = "intrinsics"
    EXTRINSICS = "extrinsics"
    AUG_TRAFO = "aug_trafo"
    EGO_POSE = "ego_pose"
    CAMERA_IDS = "camera_ids"
    SEGMENTATION = "segmentation"
    INSTANCE = "instance"
    CENTERNESS = "centerness"
    OFFSET = "offset"
    FLOW = "flow"
    FUTURE_EGOMOTION = "future_egomotion"
    Z_POSITION = "z_position"
    ATTRIBUTE = "attribute"
    BOX_2D = "box_2d"
    BOX_2D_ORIG = "box_2d_orig"
    BOX_3D = "box_3d"
    BOX_3D_ORIG = "box_3d_orig"
    BOX_3D_COMMON = "box_3d_common"
    INSTANCE_BOXES_3D = "box_3d_anno"
    INSTANCE_BOXES_2D = "box_2d_anno"
    INSTANCE_ID = "instance_id"
    HEADS = "head_labels"
    INSTANCE_SEG_2D = "instance_segmentation_2d"

    # Flags to inform about labeling state of a batch
    HAS_INSTANCE_BOXES_2D = "has_box_2d_anno"
    HAS_INSTANCE_BOXES_3D = "has_box_3d_anno"
    HAS_POINT_CLOUD = "has_point_cloud"
    HAS_OCCUPANCY = "has_occupancy"
    HAS_BEV_SURFACE = "has_bev_surface"
    HAS_INSTANCE_SEG_2D = "has_instance_segmentation_2d_anno"


@unique
class IMG_PARAMS_KEYS(str, Enum):
    FINAL_DIMS = "final_dims"
    ORIG_DIMS = "original_dims"
    SCALE_WIDTH = "scale_width"
    SCALE_HEIGHT = "scale_height"
    RESIZE_DIMS = "resize_dims"
    RESIZE_SCALE = "resize_scale"
    CROP = "crop"


@unique
class RESIZE_DIMS_INDICES(int, Enum):
    WIDTH = 0
    HEIGHT = 1


@unique
class DECODER_NAME_KEYS(str, Enum):
    RESNET18 = "resnet18"
    RESNET3D18 = "resnet3d18"
    RESNET18NECK = "resnet18neck"
    UPSAMPLING = "upsampling"
    SIMPLE = "simple"
    SIMPLE_3D = "simple_3d"
    BASIC_BLOCK_3D = "basic_block_3d"


@unique
class PRED_TASK_KEYS(str, Enum):
    SEGMENTATION = "segmentation"
    BOX_2D = "box_2d"
    BOX_3D = "box_3d"
    BOX_3D_COMMON = "box_3d_common"
    INSTANCE_FLOW = "instance_flow"
    INSTANCE_HEIGHT = "instance_height"
    INSTANCE_CENTER = "instance_center"
    INSTANCE_OFFSET = "instance_offset"
    DEPTH = "depth"
    OCCUPANCY = "occupancy"
    BEV_SURFACE = "bev_surface"
    INSTANCE_SEG_2D = "instance_segmentation_2d"
    SEGMENTATION_2D = "segmentation_2d"
    INSTANCE_CENTER_2D = "instance_center_2d"
    INSTANCE_OFFSET_2D = "instance_offset_2d"


@unique
class CAMERA_MODEL_KEYS(str, Enum):
    PINHOLE = "pinhole"
    PINHOLE_ZOOM = "pinhole_zoom"
    CYLINDRICAL = "cylindrical"
    CYLINDRICAL_ZOOM = "cylindrical_zoom"
    SPHERICAL = "spherical"


CAMERA_TYPES = {
    CAMERA_MODEL_KEYS.PINHOLE: PinholeCameraModel,
    CAMERA_MODEL_KEYS.PINHOLE_ZOOM: PinholeCameraZoomModel,
    CAMERA_MODEL_KEYS.CYLINDRICAL: CylindricalCameraModel,
    CAMERA_MODEL_KEYS.CYLINDRICAL_ZOOM: CylindricalCameraZoomModel,
    CAMERA_MODEL_KEYS.SPHERICAL: SphericalCameraModel,
}


@unique
class INTRINSIC_INDICES(int, Enum):
    FOCAL_LENGTH_X = 0
    FOCAL_LENGTH_Y = 1
    PRINCIPAL_POINT_X = 2
    PRINCIPAL_POINT_Y = 3
    SKEW = 4


# Visibility definition of nuScenes
#   1 = visibility of whole object is between 0 and 40%
#   2 = visibility of whole object is between 40 and 60%
#   3 = visibility of whole object is between 60 and 80%
#   4 = visibility of whole object is between 80 and 100%


@unique
class LH5_3D_BB_IDS(int, Enum):
    YAW_ANGLE = 0  # around z-axis
    CENTER_X = 1
    CENTER_Y = 2
    BOTTOM_Z = 3
    LENGTH = 4
    WIDTH = 5
    HEIGHT = 6
    VEL_X = 7
    VEL_Y = 8


# Box representation is explained in this image https://github.com/Dtananaev/lidar_dynamic_objects_detection/blob/3b8f3d5fcce0fb914bb83e5d43a3ca652739139e/pictures/box_parametrization.png
@unique
class NET_OUT_3D_BB_IDS(int, Enum):
    OBJECTNESS = 0
    CENTER_X = 1
    CENTER_Y = 2
    ORIENT_X = 3
    ORIENT_Y = 4
    CENTER_Z = 5
    WIDTH = 6
    HEIGHT = 7
    CLASSES_START = 8


@unique
class LABEL_MAPPING_CONFIG_KEYS(str, Enum):
    NUSCENES = "nuscenes"
    L4 = "l4"
    DST = "dst"
    KSS = "kss"