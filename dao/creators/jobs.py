from dao.models.jobs import Job, JobStatus
import dao.consts.outputs as consts_outputs
import time


class JobCreator:

    @staticmethod
    def create_new_job(_title: str, initial_outputs: list = None):
        if initial_outputs is None:
            initial_outputs = []

        # Adding default initial job created output.
        initial_outputs.insert(0, {
            consts_outputs.KEY_OUTPUT_TYPE: consts_outputs.VALUE_OUTPUT_TYPE_JOB_CREATED,
            consts_outputs.KEY_TIMESTAMP: time.time(),
        })

        job: Job = Job(title=_title, status=JobStatus.CREATED, outputs=initial_outputs)
        job.save()

        return job.objectId
