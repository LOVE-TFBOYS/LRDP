from .flow_utils import compose_flows, detach_flow, identity_grid_3d, resize_flow, upsample_flow
from .initial_flow import InitialFlowNet, InitialFlowNet3D, VTNLite
from .recursive_refine import CNNRefineBlock
from .spatial_transformer import SpatialTransformer3D

__all__ = [
    "CNNRefineBlock",
    "InitialFlowNet",
    "InitialFlowNet3D",
    "LRDPRegistrationModel",
    "SpatialTransformer3D",
    "VTNLite",
    "build_lrdp_model",
    "compose_flows",
    "detach_flow",
    "identity_grid_3d",
    "resize_flow",
    "upsample_flow",
]


def __getattr__(name):
    if name in {"LRDPRegistrationModel", "build_lrdp_model"}:
        from .lrdp_model import LRDPRegistrationModel, build_lrdp_model

        return {"LRDPRegistrationModel": LRDPRegistrationModel, "build_lrdp_model": build_lrdp_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
