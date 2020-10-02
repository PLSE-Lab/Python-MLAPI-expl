import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
print(os.listdir("../input"))

# Leap Motion files parser
# Leap Motion file contains metadata and a sequence of frames;
# Each frame is a 3D representation of all bones in a human hand

ID = 0
TIMESTAMP = 1
HANDS = 2
POINTABLES = 3
INTERACTION_BOX = 4

# comes in first leap motion data frame
FRAME_STRUCTURE = json.loads('{"frame":'
                             '["id", "timestamp", {"hands": [["id", "type", "direction", "palmNormal", '
                             '"palmPosition", "palmVelocity", "stabilizedPalmPosition", "pinchStrength", '
                             '"grabStrength", "confidence", "armBasis", "armWidth", "elbow", "wrist"]]}, '
                             '{"pointables": [["id", "direction", "handId", "length", "stabilizedTipPosition", '
                             '"tipPosition", "tipVelocity", "tool", "carpPosition", "mcpPosition", "pipPosition", '
                             '"dipPosition", "btipPosition", "bases", "type"]]}, {"interactionBox": ["center", '
                             '"size"]}]'
                             '}')
# needed to transform text file representation into a convenient format                            
class Index:
    def __init__(self, frame_str=FRAME_STRUCTURE["frame"]):
        hands_ = frame_str[HANDS]["hands"][0]
        self.hand_item_index = self.index_json(hands_)
        self.hand_index = self.reverse(self.hand_item_index)

        pointables_ = frame_str[POINTABLES]["pointables"][0]
        self.pointables_item_index = self.index_json(pointables_)
        self.pointables_index = self.reverse(self.pointables_item_index)

        self.int_box_item_index = self.index_json(frame_str[INTERACTION_BOX]["interactionBox"])
        self.int_box_index = self.reverse(self.int_box_item_index)

    @staticmethod
    def index_json(frame_str):
        return dict(enumerate(frame_str))

    @staticmethod
    def reverse(index_to_name):
        return {v: k for k, v in index_to_name.items()}


index = Index()


def get_string_template(*args):
    r = range(len(args))
    templ = str(["{{}}".format(x) for x in r])
    formatted = templ.format(*args).replace("'{}'", "{}")[2:-2]\
        .replace("False", "false")\
        .replace("True", "true")
    return formatted


class InteractionBox:
    def __init__(self, json_data=None, interaction_box=None, center=None, size=None):
        if None is not json_data:
            self.center = json_data[index.int_box_index["center"]]
            self.size = json_data[index.int_box_index["size"]]
        else:
            self.center = center
            self.size = size

    def __getitem__(self, key):
        return getattr(self, index.int_box_item_index[key])

    def __str__(self):
        return get_string_template(list(map(lambda x: self[x], range(0, len(index.int_box_index)))))


class Pointables:
    def __init__(self, json_data=None, pointables=None):
        if None is not json_data:
            self.pointables = list(map(lambda j: Pointable(json_data=j), json_data))
        else:
            self.pointables = pointables

    def __str__(self):
        res = "["
        for p in self.pointables:
            res = res + str(p) + ", "
        return res[:-2]+"]"

# these are fingers
class Pointable:
    def __init__(self, json_data=None, id=None, direction=None, hand_id=None, length=None, stabilized_tip_position=None,
                 tip_position=None, tip_velocity=None, tool=None, carp_position=None, mcp_position=None, pip_position=None,
                 dip_position=None, btip_position=None, bases=None, pointable_type=None):
        if None is not json_data:
            self.id = json_data[index.pointables_index["id"]]
            self.direction = json_data[index.pointables_index["direction"]]
            self.handId = json_data[index.pointables_index["handId"]]
            self.length = json_data[index.pointables_index["length"]]
            self.stabilizedTipPosition = json_data[index.pointables_index["stabilizedTipPosition"]]
            self.tipPosition = json_data[index.pointables_index["tipPosition"]]
            self.tipVelocity = json_data[index.pointables_index["tipVelocity"]]
            self.tool = json_data[index.pointables_index["tool"]]
            self.carpPosition = json_data[index.pointables_index["carpPosition"]]
            self.mcpPosition = json_data[index.pointables_index["mcpPosition"]]
            self.pipPosition = json_data[index.pointables_index["pipPosition"]]
            self.dipPosition = json_data[index.pointables_index["dipPosition"]]
            self.btipPosition = json_data[index.pointables_index["btipPosition"]]
            self.bases = json_data[index.pointables_index["bases"]]
            self.type = json_data[index.pointables_index["type"]]
        else:
            self.id = id
            self.direction = direction
            self.hand_id = hand_id
            self.length = length
            self.stabilizedTipPosition = stabilized_tip_position
            self.tipPosition = tip_position
            self.tipVelocity = tip_velocity
            self.tool = tool
            self.carpPosition = carp_position
            self.mcpPosition = mcp_position
            self.pipPosition = pip_position
            self.dipPosition = dip_position
            self.btipPosition = btip_position
            self.bases = bases
            self.type = pointable_type

    def __getitem__(self, key):
        return getattr(self, index.pointables_item_index[key])

    def __str__(self):
        res = get_string_template(list(map(lambda x: self[x], range(0, len(index.pointables_item_index)))))
        return res


class Hands:
    def __init__(self, json_data=None, hands=None):
        if None is not json_data:
            self.hands = list(map(lambda j: Hand(json_data=j), json_data))
        else:
            self.hands = hands

    def __str__(self):
        res = "["
        for hand in self.hands:
            res = res + str(hand) + ", "
        return res[:-2]+"]"


class Hand:
    def __init__(self, json_data=None, id=None, type=None, direction=None, palm_normal=None, palm_position=None,
                 palm_velocity=None, stabilized_palm_position=None, pinch_strength=None,
                 grab_strengt=None, confidence=None, arm_basis=None, arm_width=None, elbow=None, wrist=None):
        if None is not json_data:
            self.id = json_data[index.hand_index["id"]]
            self.type = json_data[index.hand_index["type"]]
            self.direction = json_data[index.hand_index["direction"]]
            self.palmNormal = json_data[index.hand_index["palmNormal"]]
            self.palmPosition = json_data[index.hand_index["palmPosition"]]
            self.palmVelocity = json_data[index.hand_index["palmVelocity"]]
            self.stabilizedPalmPosition = json_data[index.hand_index["stabilizedPalmPosition"]]
            self.pinchStrength = json_data[index.hand_index["pinchStrength"]]
            self.grabStrength = json_data[index.hand_index["grabStrength"]]
            self.confidence = json_data[index.hand_index["confidence"]]
            self.armBasis = json_data[index.hand_index["armBasis"]]
            self.armWidth = json_data[index.hand_index["armWidth"]]
            self.elbow = json_data[index.hand_index["elbow"]]
            self.wrist = json_data[index.hand_index["wrist"]]
        else:
            self.id = id
            self.type = type
            self.direction = direction
            self.palmNormal = palm_normal
            self.palmPosition = palm_position
            self.palmVelocity = palm_velocity
            self.stabilizedPalmPosition = stabilized_palm_position
            self.pinchStrength = pinch_strength
            self.grabStrength = grab_strengt
            self.confidence = confidence
            self.armBasis = arm_basis
            self.armWidth = arm_width
            self.elbow = elbow
            self.wrist = wrist

    def __getitem__(self, key):
        return getattr(self, index.hand_item_index[key])

    def __str__(self):
        res = get_string_template(list(map(lambda x: self[x], range(0, len(index.hand_item_index)))))
        return res

#Leap Motion SDK 2 format
class NativeFrame:
    def __init__(self, LeapFrame):
        self.id = LeapFrame.id
        self.timestamp = LeapFrame.timestamp
        self.hands = LeapFrame.hands.hands
        self.pointables = LeapFrame.pointables.pointables
        self.interactionBox = LeapFrame.interactionBox


class LeapFrame:
    def __init__(self, str_data=None, json_data=None, id=None, timestamp=None, hands=None, pointables=None,
                 interaction_box=None):
        if None is not str_data:
            json_data = json.loads('{"x":' + str_data + '}')['x']
        if None is not json_data:
            self.hands = Hands(json_data[HANDS])
            self.pointables = Pointables(json_data[POINTABLES])
            self.interactionBox = InteractionBox(json_data[INTERACTION_BOX])
            self.id = json_data[ID]
            self.timestamp = json_data[TIMESTAMP]
        else:
            self.hands = hands
            self.pointables = pointables
            self.interactionBox = interaction_box
            self.id = id
            self.timestamp = timestamp

    def __str__(self):
        return '[{}, {}, {}, {}, {}]'\
            .format(self.id, self.timestamp, self.hands, self.pointables, self.interactionBox)\
            .replace("'", '"').replace('None', 'null')

    def to_json(self):
        return json.dumps(NativeFrame(self), default=lambda o: o.__dict__,
            sort_keys=False, indent=None)
            

def read_file(name='../input/50sec1.json'): return open(name, "r").read()

leap_data = json.loads(read_file())

frames = leap_data['frames']
#frame #0 is not actually hand recording data
metadata = frames[0]
print('Loaded frames from a file, here is one: ' + str(frames[1]))

#parse a random frame:
frame = LeapFrame(json_data=frames[77])
print('Parsed a leap frame #77: ' + str(frame.to_json()))
