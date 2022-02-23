from parse_rest.datatypes import Object


class Job(Object):
    pass


class JobStatus:

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    ONGOING = "ONGOING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
