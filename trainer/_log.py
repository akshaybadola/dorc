import inspect


class Log:
    def __init__(self, logger, production=False):
        self.logger = logger
        self.production = production

    def _logd(self, x):
        "Log to DEBUG and return string with name of calling function"
        if self.production:
            self.logger.debug(x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.debug(x)
            return x

    def _loge(self, x):
        "Log to ERROR and return string with name of calling function"
        if self.production:
            self.logger.error(x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.error(x)
            return x

    def _logi(self, x):
        "Log to INFO and return string with name of calling function"
        if self.production:
            self.logger.info(x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.info(x)
            return x

    def _logw(self, x):
        "Log to WARN and return string with name of calling function"
        if self.production:
            self.logger.warn(x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.warning(x)
            return x
