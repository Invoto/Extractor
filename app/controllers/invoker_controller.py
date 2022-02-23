from fastapi import UploadFile
from wrappers.files import LocalFile
from dao.wrappers.jobs import JobHandlerWrapper
from dao.models.jobs import JobStatus
import app.consts.responses as consts_responses
import services.fetchers as service_fetches
from services.abstracts import BasicService
from services.impl.enqueuer import EnqueuerService


def get_invoke_response(upload_invoice_file: UploadFile):
    unique_local_file_name = LocalFile.get_unique_local_file_name(upload_invoice_file.filename)
    invoice_file: LocalFile = LocalFile(unique_local_file_name, spooled_file=upload_invoice_file.file)

    wrapper_job: JobHandlerWrapper = JobHandlerWrapper()
    job_id: str = wrapper_job.get_job_id()

    # Enqueuing the job.
    enqueuer_service: BasicService = service_fetches.fetch_by_name(EnqueuerService.SERVICE_NAME)
    enqueuer_service.request(job_id, invoice_file.get_abs_file_path())

    # Updating Job Status.
    wrapper_job.update_job_status(JobStatus.QUEUED)
    wrapper_job.progress_update("Queued.")

    return {
        consts_responses.KEY_STATUS: True,
        consts_responses.KEY_JOB_TASK_JOB_ID: job_id,
    }
