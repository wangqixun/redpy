from .server.instance_segm.client import Client as InstanceSegmClient
# from .server.depth_estimation.client import Client as DepthEstimationClient
from .server.sky_segmentation.client import Client as SkySegmentationClient
from .server.salient_object_segm.client_gprc import Client as SalientObjectSegmClient
from .server.diffusion.client import Client as DiffusionClient
from .server.translate.client import Client as TranslateClient
from .server.common.client import Client as CommonClient



__all__ = [
    'InstanceSegmClient', 'SkySegmentationClient', 'SalientObjectSegmClient',
    'DiffusionClient', 'TranslateClient', 'CommonClient'
]











