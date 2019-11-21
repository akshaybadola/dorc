class Tag:
    def __init__(self, x):
        self.tag = x
        self._members = []

    @property
    def members(self):
        return self._members

    def __call__(self, f):
        # if self.tag not in f.__dict__:
        #     f.__dict__[self.tag] = True
        #     self._funcs.append(f)
        # if callable(f):
        #     if f not in self._members:
        #         self._members.append(f)
        #     return f
        # else:
        #     self._members.append(f)
        if f not in self._members:
            self._members.append(f)
        return f


control = Tag("control")
prop = Tag("prop")
