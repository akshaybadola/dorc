import inspect


class Log:
    def __init__(self, logger, production=False):
        self.logger = logger
        self.production = production

    def _logd(self, x):
        "Log to DEBUG and return string with name of calling function"
        if self.production:
            self.logger.debug("[DEBUG]" + x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.debug("[DEBUG]" + x)
            return x

    def _loge(self, x):
        "Log to ERROR and return string with name of calling function"
        if self.production:
            self.logger.error("[ERROR]" + x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.error("[ERROR]" + x)
            return x

    def _logi(self, x):
        "Log to INFO and return string with name of calling function"
        if self.production:
            self.logger.info("[INFO]" + x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.info("[INFO]" + x)
            return x

    def _logw(self, x):
        "Log to WARN and return string with name of calling function"
        if self.production:
            self.logger.warning("[WARNING]" + x)
            return x
        else:
            f = inspect.currentframe()
            prev_func = inspect.getframeinfo(f.f_back).function
            x = f"[{prev_func}()] " + x
            self.logger.warning("[WARNING]" + x)
            return x
