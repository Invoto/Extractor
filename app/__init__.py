from fastapi import FastAPI
from . import routes
import services


application = FastAPI()

for route in routes.ALL_ROUTES:
    application.include_router(route.router, prefix="/" + route.PREFIX, tags=route.TAGS)


# Startup Events - When application starts up.
@application.on_event("startup")
async def before_startup():
    services.start_services()


# Shutdown Events - When application shuts down.
@application.on_event("shutdown")
async def at_shutdown():
    services.stop_services()
