import datetime
import hashlib
import flask_login


class User(flask_login.UserMixin):
    def __init__(self, id, name):
        self.id = id
        self.name = name

    # def get_id(self):
    #     hashlib.sha1(datetime.datetime.now().isoformat())

    def __repr__(self):
        return f"{self.id}_{self.name}"


def __inti__(_n):
    if _n == hashlib.sha1(("2ads;fj4sak#)" + "admin").encode("utf-8")).hexdigest():
        return "AdminAdmin_33"
    elif _n == hashlib.sha1(("2ads;fj4sak#)" + "joe").encode("utf-8")).hexdigest():
        return "Monkey$20"
    elif _n == hashlib.sha1(("2ads;fj4sak#)" + "taruna").encode("utf-8")).hexdigest():
        return "Donkey_02"
    else:
        return None


__ids__ = {0: "admin", 1: "joe", 2: "taruna"}
__users__ = {"admin": User(0, "admin"),
             "joe": User(1, "joe"),
             "taruna": User(2, "taruna")}
