class StateMachine:
    """Each state is a string delimited by _. State string size has to be consistent."""
    def __init__(self, state_length, steps, **kwargs):
        self._state_length = state_length
        self._transition_steps = steps
        self._logd = kwargs["logd"]
        self._loge = kwargs["loge"]
        self._logi = kwargs["logi"]
        self._logw = kwargs["logw"]
        self._init_predicates()
        self._current_state = "normal_paused_none"

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, x):
        self._current_state = x

    def legal_states(self, a, b):
        a_force, a_run, a_step = a.split("_")
        b_force, b_run, b_step = b.split("_")
        return (a_force in {"force", "normal"} and b_force in {"force", "normal"}
                and a_run in {"paused", "running", "finished"}
                and b_run in {"paused", "running", "finished"})

    def _init_predicates(self):
        def normal_to_normal_same_step(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "normal" == b_force
                    and a_step == b_step  # in the same step
                    and ((a_run != b_run and a_run in {"running", "paused"}
                          and b_run in {"running", "paused"})  # running <-> paused
                         # only paused -> finished
                         or (a_run == "paused" and b_run == "finished")))

        def normal_to_norml_different_step(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "normal" == b_force
                    and ((a_step != b_step
                          and a_run == "finished"  # finished -> {running, paused}
                          and b_run in {"running", "paused"})
                         or (a_step == "none" and a_run == "paused"  # if none -> any
                             and b_run in {"running", "paused"})))

        # CHECK: Can this happen? 
        def normal_to_force_same_step(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "normal" != b_force  # normal -> force
                    and a_step == b_step            # same_step
                    and ((a_run in {"paused", "running"}
                          and b_run in {"paused", "running"})  # paused <-> running
                         or (a_run == "paused"
                             and b_run == "finished")))  # paused -> finished

        def normal_to_force_different_step(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "normal" != b_force        # normal -> force
                    and a_step != b_step
                    # no point a_running -> b_paused
                    and b_run == "running"
                    and a_run in {"paused", "finished"})

        def normal_to_force_stop(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "normal" != b_force        # normal -> force
                    and b_step == "stop"
                    and a_run == "paused"  # paused -> finished only?
                    and b_run == "finished")  # if finished it must be in forced state

        def force_to_normal(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "force" != b_force
                    # paused previous first
                    and a_run == "finished" and b_run in {"paused", "running"})

        def force_to_force(a, b):
            a_force, a_run, a_step = a.split("_")
            b_force, b_run, b_step = b.split("_")
            return (a_force == "force" == b_force)

        self._transition_predicates = [normal_to_normal_same_step,
                                       normal_to_norml_different_step,
                                       normal_to_force_same_step,
                                       normal_to_force_different_step,
                                       normal_to_force_stop,
                                       force_to_normal,
                                       force_to_force]

    def allowed_transition(self, a, b, debug=False):
        if not len(a.split("_")) == len(b.split("_")) == self._state_length:
            return False
        if not self.legal_states(a, b):
            return False
        if b == "normal_running_none":
            return False
        for p in self._transition_predicates:
            if p(a, b):
                if debug:
                    self._logd(f"{p.__name__} was True")
                return True
        return False
