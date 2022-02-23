from services.abstracts import BasicService
from parse_rest.connection import register
import config.environment as config_env


class ParseConnectorService(BasicService):

    SERVICE_NAME = "ParseConnectorService"

    def get_name(self):
        return self.SERVICE_NAME

    def run(self):
        register(config_env.get_parse_app_id(), config_env.get_parse_api_key(), master_key=config_env.get_parse_master_key())

    def stop(self):
        pass

    def request(self, *args):
        pass
