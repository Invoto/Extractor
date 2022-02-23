from dao.managers.jobs import JobManager
from dao.creators.jobs import JobCreator
from dao.models.jobs import JobStatus
import dao.consts.outputs as consts_outputs
import time


class JobHandlerWrapper:

    JOB_IDENTIFIER = "invoker_invoice"

    def __init__(self, job_id: str = None):
        self._m_job_id: str = job_id

        if self._m_job_id is None:
            self._create()

    def get_job_id(self):
        return self._m_job_id

    def get_reserved_job_manager(self):
        return JobManager(self._m_job_id)

    def _create(self):
        self._m_job_id = JobCreator.create_new_job(JobHandlerWrapper.JOB_IDENTIFIER)

    def remove(self):
        job_manager: JobManager = self.get_reserved_job_manager()
        job_manager.remove_job()

    def update_job_status(self, job_status: str):
        self.get_reserved_job_manager().update_job_status(job_status)

    def progress_update(self, message: str):
        job_manager = self.get_reserved_job_manager()
        job_manager.append_output({
            consts_outputs.KEY_OUTPUT_TYPE: consts_outputs.VALUE_OUTPUT_TYPE_PROGRESS_UPDATE,
            consts_outputs.KEY_TIMESTAMP: time.time(),
            consts_outputs.KEY_MESSAGE: message,
        })

    def complete(self, entities: dict):
        # Update Progress first
        self.progress_update("Job Completed.")

        job_manager: JobManager = self.get_reserved_job_manager()

        # Updates job status.
        job_manager.append_output({
            consts_outputs.KEY_OUTPUT_TYPE: consts_outputs.VALUE_OUTPUT_TYPE_RESULT,
            consts_outputs.KEY_STATUS: True,
            **entities
        })

        # Mark the job as completed here.
        job_manager.update_job_status(JobStatus.COMPLETED)

    def fail(self, failure_message: str):
        # Update Progress first
        self.progress_update(failure_message)

        # Retrieve a Job Manager.
        job_manager: JobManager = self.get_reserved_job_manager()

        # Updates job status.
        job_manager.append_output({
            consts_outputs.KEY_OUTPUT_TYPE: consts_outputs.VALUE_OUTPUT_TYPE_RESULT,
            consts_outputs.KEY_STATUS: False,
            consts_outputs.KEY_MESSAGE: failure_message,
        })

        # Mark the job as completed here.
        job_manager.update_job_status(JobStatus.FAILED)
