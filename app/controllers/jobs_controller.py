from dao.managers.jobs import JobManager
import app.consts.responses as consts_responses


def get_job(job_id: str):
    job_manager: JobManager = JobManager(job_id)
    job_exists, job = job_manager.get_job_exists_and_job()

    if job_exists:
        return {
            consts_responses.KEY_STATUS: True,
            "id": job.objectId,
            "title": job.title,
            "status": job.status,
            "outputs": job.outputs,
            "created_at": job.createdAt,
            "updated_at": job.updatedAt,
        }
    else:
        return {
            consts_responses.KEY_STATUS: False,
            consts_responses.KEY_FAILURE_MESSAGE: "No such Job.",
        }
