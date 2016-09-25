import os


class Configuration(object):

    def get_datapath(self, dataset=None):
        if dataset is None:
            return os.environ.get("PYTHON_DATA_FOLDER", "../data/")
        env_variable = "PYTHON_DATA_FOLDER_%s" % dataset.upper()
        return os.environ.get(env_variable, "/home/jogi/git/repository/smart_play_set/data")


config = Configuration()
