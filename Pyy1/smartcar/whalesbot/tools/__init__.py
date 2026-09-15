from .tools_class import *
from .camera import Camera
from .streamer import Streamer
from .log_wrap import logger
# CollectControlCar 采用懒加载，避免循环导入：
#   mc601_ctl2 → tools.log_wrap → tools.__init__ → collect_control → vehicle (循环!)
def __getattr__(name):
    if name == 'CollectControlCar':
        from .collect_control import CollectControlCar
        return CollectControlCar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

