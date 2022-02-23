from tempfile import SpooledTemporaryFile
import config.paths as config_paths
import os
import time


class LocalFile:

    def __init__(self, file_name: str, spooled_file: SpooledTemporaryFile = None):
        self._m_file_name: str = file_name

        if spooled_file is not None:
            self._store_spooled_file(spooled_file)

    def get_abs_file_path(self):
        return os.path.abspath(config_paths.LOCAL_FILES_PATH + "/" + self._m_file_name)

    @staticmethod
    def get_abs_directory_path():
        return os.path.abspath(config_paths.LOCAL_FILES_PATH)

    def get_file_name(self):
        return self._m_file_name

    def _store_spooled_file(self, spooled_file: SpooledTemporaryFile):
        # Create the directory if does not exist.
        os.makedirs(os.path.dirname(self.get_abs_file_path()), exist_ok=True)
        with open(self.get_abs_file_path(), "wb") as file_to_save:
            file_to_save.write(spooled_file.read())
            file_to_save.close()

    @staticmethod
    def get_unique_local_file_name(file_name: str):
        return str(time.time()).replace(".", "") + "_" + file_name

    def _write_data(self, mode: str, data):
        # This would automatically create if the file does not exist.
        with open(self.get_abs_file_path(), mode) as file:
            file.write(data)
            file.close()

    def write_binary_data(self, data, append=False):
        return self._write_data("wb" if not append else "ab", data)

    def write_text_data(self, data: str, append=False):
        return self._write_data("w" if not append else "a", data)

    def delete(self):
        os.remove(self.get_abs_file_path())
