from fastapi import APIRouter
import app.controllers.jobs_controller as controller_jobs


# Prefix for jobs endpoints.
PREFIX = "jobs"
# Tags for docs.
TAGS = ["Jobs"]

router = APIRouter()


@router.get("/")
async def index():
    return {
        "message": PREFIX.upper(),
    }


@router.get("/{job_id}")
async def get_job(job_id: str):
    return controller_jobs.get_job(job_id)
