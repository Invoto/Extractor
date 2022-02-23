from dao.models.jobs import Job, JobStatus
from parse_rest.query import QueryResourceDoesNotExist


class JobManager:

    def __init__(self, job_id: str):
        self._m_job_id: str = job_id

    def _get_job(self):
        try:
            return Job.Query.get(objectId=self._m_job_id)

        except QueryResourceDoesNotExist:
            return None

    def get_job_exists_and_job(self):
        job: Job = self._get_job()
        return (True, job) if job is not None else (False, None)

    def update_job_status(self, status: str):
        job_exists, job = self.get_job_exists_and_job()
        if job_exists:
            job.status = status
            job.save()

        return job_exists

    def append_output(self, output: dict):
        job_exists, job = self.get_job_exists_and_job()
        if job_exists:
            job.outputs.append(output)
            job.save()

        return job_exists

    def remove_job(self):
        job_exists, job = self.get_job_exists_and_job()
        if job_exists:
            job.delete()

        return job_exists
