from types import SimpleNamespace


class Config(SimpleNamespace):
    """Stateless configuration of the trainer"""
    def __init__(self, config):
        super().__init__()
        assert {"model_params", "criteria_params", "dataloader_params", "training_params"}\
            == set(config.keys())
        self.__dict__.update(config)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __iter__(self):
        return self.__dict__.__iter__()

    def __getitem__(self, x):
        self.__dict__[x]

    def __setitem__(self, x, y):
        self.__dict__[x] = y

    def __len__(self):
        return len(self.__dict__)


    
