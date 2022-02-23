import services


def fetch_by_name(service_name: str):
    for service in services.ALL_SERVICES:
        if service.get_name() == service_name:
            return service

    return None
