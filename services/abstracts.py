import abc


class BasicService(abc.ABC):

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def request(self, *args):
        pass
