class DatasetConfig(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def __getattr__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        return self.__class__.__getattr__(item)

