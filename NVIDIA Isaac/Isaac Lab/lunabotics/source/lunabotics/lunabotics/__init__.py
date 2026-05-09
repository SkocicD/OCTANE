"""CSU Lunabotics Isaac Lab extension — registers Gym environments and UI extensions."""

try:
    from .tasks import *
except ModuleNotFoundError:
    pass  # Isaac Lab not available (e.g. standalone unit tests)
