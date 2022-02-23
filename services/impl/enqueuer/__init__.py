from services.abstracts import BasicService
import config.environment as config_env
from services.impl.enqueuer import executor
from multiprocessing import Process


class EnqueuerService(BasicService):

    SERVICE_NAME = "EnqueuerService"

    def __init__(self):
        self._m_all_processes = []

    def get_name(self):
        return self.SERVICE_NAME

    def run(self):
        pass

    def stop(self):
        for process in self._m_all_processes:
            process.join()

    def request(self, job_id: str, invoice_file_path: str):
        model_chkpt_path: str = config_env.get_model_chkpt_path()

        process: Process = Process(target=executor.predict_image, kwargs={
            "job_id": job_id,
            "image_file_path": invoice_file_path,
            "model_chkpt_path": model_chkpt_path,
            "parse_app_id": config_env.get_parse_app_id(),
            "parse_api_key": config_env.get_parse_api_key(),
            "parse_master_key": config_env.get_parse_master_key(),
        })

        self._m_all_processes.append(process)

        process.start()
