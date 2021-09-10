import io
import pickle
import torch


# Helper methods for easier changing of byte conversion
def serialize(obj):
    return pickle.dumps(obj)


def unserialize(obj):
    return pickle.loads(obj)


def serialize_model(mdl):
    buf = io.BytesIO()
    torch.save([mdl.actor, mdl.critic, mdl.shared], buf)
    return buf


def unserialize_model(buf):
    return torch.load(buf)
