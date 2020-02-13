class StateMachine:
    """Each state is a string delimited by _. State string size has to be consistent."""
    def __init__(self, state_length, steps, **kwargs):
        self._state_length = state_length
        self._transition_steps = steps
        self._logd = kwargs["logd"]
        self._loge = kwargs["loge"]
        self._logi = kwargs["logi"]
        self._logw = kwargs["logw"]
        self._current_state = "main_paused_none"

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, x):
        self._current_state = x

    def legal_states(self, a, b):
        a_loop, a_run, a_step = a.split("_")
        b_loop, b_run, b_step = b.split("_")
        return (a_loop in {"main", "adhoc", "user"} and b_loop in {"main", "adhoc", "user"}
                and a_run in {"paused", "running", "finished"}
                and b_run in {"paused", "running", "finished"})

    def allowed_transition(self, a, b, debug=False):
        if not len(a.split("_")) == len(b.split("_")) == self._state_length:
            return False
        if not self.legal_states(a, b):
            return False
        if b.endswith("_running_none"):
            return False
        for p in self._transition_predicates:
            if p(a, b):
                if debug:
                    self._logd(f"{p.__name__} was True")
                return True
        return False
