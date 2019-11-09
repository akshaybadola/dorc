class Tag:
    def __init__(self, x):
        self.tag = x
        self._funcs = []

    def __call__(self, f):
        # if self.tag not in f.__dict__:
        #     f.__dict__[self.tag] = True
        #     self._funcs.append(f)
        if f not in self._funcs:
            self._funcs.append(f)
        return f


control = Tag("control")
