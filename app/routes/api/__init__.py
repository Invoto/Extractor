from fastapi import APIRouter
from . import invoker as routes_invoker
from . import jobs as routes_jobs

# The prefix for API.
PREFIX = "api"
# Tags for docs.
TAGS = ["API"]

# Declaring the API router and including sub-routes
router = APIRouter()
router.include_router(routes_invoker.router, prefix="/" + routes_invoker.PREFIX, tags=routes_invoker.TAGS)
router.include_router(routes_jobs.router, prefix="/" + routes_jobs.PREFIX, tags=routes_jobs.TAGS)


@router.get("/")
async def index():
    return {
        "message": PREFIX.upper(),
    }
