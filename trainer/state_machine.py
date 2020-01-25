class StateMachine:
    def __init__(self, current_state, steps, forced_states, logd, loge, logi, logw):
        self._current_state = current_state
        self._transition_steps = steps
        self._forced_states = forced_states
        self._logd = logd
        self._loge = loge
        self._logi = logi
        self._logw = logw

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, x):
        self._current_state = x

    def allowed_transition(self, a, b):
        a_force, a_run, a_step = a.split("_")
        b_force, b_run, b_step = b.split("_")
        # if a_run == b_run == "paused":  # can we have normal_paused_train -> normal_paused_val?
        #     return False
        # if a_force == b_force == "force":
        #     return False
        allowed_predicate_list = [(a_force == "normal" == b_force
                                   and a_step == b_step  # in the same step
                                   and ((a_run != b_run and a_run in {"running", "paused"}
                                         and b_run in {"running", "paused"})  # running <-> paused
                                        # only running -> finished
                                        or (a_run == "running" and b_run == "finished"))),
                                  (a_force == "normal" == b_force
                                   and ((a_step != b_step
                                         and a_run == "finished"  # finished -> {running, paused}
                                         and b_run in {"running", "paused"})
                                        or (a_step == "none" and a_run == "paused"  # if none -> any
                                            and b_run in {"running", "paused"}))),  # paused -> {running, paused}
                                  (a_force == "normal" != b_force        # normal -> force
                                   and a_run != "running"
                                   and b_run == "running"  # any -> running
                                   and a_step != b_step
                                   and a_step not in self._forced_states
                                   and b_step in self._forced_states),  # step must change
                                  (a_force == "force" != b_force
                                   # paused previous first
                                   and a_run == "finished" and b_run in {"paused", "running"}
                                   and a_step in self._forced_states
                                   and b_step not in self._forced_states)]
        if any(allowed_predicate_list):
            return True
        else:
            return False
