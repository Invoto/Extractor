from .parse import ParseConnectorService
from .enqueuer import EnqueuerService

ALL_SERVICES = [
    ParseConnectorService(),
    EnqueuerService(),
]
