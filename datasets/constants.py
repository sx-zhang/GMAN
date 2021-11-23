KITCHEN_OBJECT_CLASS_LIST = [
    "Microwave",
    "Fridge",
    "GarbageCan",
    "Bowl",
    "Cabinet",
    'Pan',
    'Toaster'
]

LIVING_ROOM_OBJECT_CLASS_LIST = [
    "Pillow",
    "Television",
    "GarbageCan",
    'FloorLamp',
    'Shelf',
    'Laptop',
    'Sofa',
    'Box',
    'SideTable',
]

BEDROOM_OBJECT_CLASS_LIST = ["Book", "AlarmClock",'Mug','Chair']


BATHROOM_OBJECT_CLASS_LIST = ["Sink",  'Bathtub', "LightSwitch",'ShowerCurtain']

FULL_OBJECT_CLASS_LIST = (
    KITCHEN_OBJECT_CLASS_LIST
    + LIVING_ROOM_OBJECT_CLASS_LIST
    + BEDROOM_OBJECT_CLASS_LIST
    + BATHROOM_OBJECT_CLASS_LIST
)


MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_UP = "LookUp"
LOOK_DOWN = "LookDown"
DONE = "Done"

DONE_ACTION_INT = 5
GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01
