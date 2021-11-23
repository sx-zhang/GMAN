class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(
        self, state=None, hidden=None, target_class_embedding=None, action_probs=None, target_object=None,
            att_in_view=None, room_name=None
    ):
        self.state = state
        self.hidden = hidden
        self.target_class_embedding = target_class_embedding
        self.action_probs = action_probs
        self.target_object = target_object
        self.att_in_view = att_in_view
        self.room_name = room_name


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None, fake_img=None):

        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
        self.fake_img = fake_img
