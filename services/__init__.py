from .impl import ALL_SERVICES


def start_services():
    for service in ALL_SERVICES:
        service.run()


def stop_services():
    for service in ALL_SERVICES:
        service.stop()
