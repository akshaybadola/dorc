class CatchAll:
    def __init__(self, success_message, failure_message, checks):
        self.success_message = success_message
        self.failure_message = failure_message
        self._checks = checks

    def __enter__(self):
        return True

    def __exit__(self, exc_type, exc_msg, tb):
        if exc_type:
            self._checks.status = False
            self._checks.message = self.failure_message + f" Error occured." +\
                f" {exc_type.__qualname__}: {exc_msg}"
            self._checks.message_func = self._checks._error_func
        else:
            self._checks.status = True
            self._checks.message = self.success_message
            self._checks.message_func = self._checks._message_func
        return True


class Failure:
    def __enter__(self):
        return False

    def __exit__(self, *exc_info):
        pass


class Checks:
    def __init__(self, default_message_func, default_error_func):
        self._list = []
        self._message_func = default_message_func
        self._error_func = default_error_func

    def clear(self):
        self._list = []
        self.status = None
        self.message = None
        self.message_func = None

    def add(self, check, error_message, func=None):
        self._list.append((check, error_message,
                           func if func else self._error_func))

    def check_all_true(self, message=None, func=None):
        for p, e, f in self._list:
            if not p:
                self.status = False
                self.message = e
                self.message_func = f
                return
        self.status = True
        self.message = "All checks passed" if not message else message
        self.message_func = self._message_func if not func else func

    def check_which_one_true(self, message=None, func=None):
        for p, m, f in self._list:
            if p:
                self.status = True
                self.message = m
                self.message_func = f
                return
        self.status = False
        self.message = "None were true" if not message else message
        self.message_func = self._error_func if not func else func

    def catch_and_log(self, success_message, failure_message):
        if self.status:
            return CatchAll(success_message, failure_message, self)
        else:
            return Failure()
