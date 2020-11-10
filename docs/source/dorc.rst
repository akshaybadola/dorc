.. _dorc-root:

DORC
====

Need
----

DORC is network centric and is designed to be framework agnostic. However, at
present only pytorch is truly integrated. The principles of model fitting are
universal. `Deep Learning` has brought about a paradigm shift in large
nonparametric statistical models, and fitting them even with standard methods is
non trivial. Couple that with the training time required and the need for
interpretation of training at multiple `layers` of the model, at times requires
an in depth assessment how the model is converging to debug any potential
issues.


Device Allocation
-----------------

Devices are allocated both to the trainer and to the models. Devices allocated
to a trainer will not be available to other training sessions. Devices within a
training sessions can be shuffled around to various models available to the
trainer as the need arises and according to the commands given.

Special directives can be given which would ensure automatic device
allocation.

- GPUs (or systems) themselves can be allocated to the trainer in
  `trainer_params`.
- After which the models themselves can have which can set which devices they're
  active on. We can use a ``_device`` attr on the model to determine which
  device to assign to which model.
- Special device names like `parallel`, `dataparallel` or
  `distributed_dataparallel` can indicate that the model is requested to be
  spread over all the possible GPUs (or systems). Otherwise device names


User Functions
--------------

1. `user_func` shouldn't be given access to the trainer instance itself as
   it may cause unwanted states
2. `hooks` may be specific functions which can be given access to specific
   locals of particular functions
3. `user_func` is the most generic function and has arbitrary call and
   return values
4. In constrast `hooks` would be more restriced and while adding or
   removing a hook certain checks can be performed
5. `user_funcs` maybe invoked in the middle of the program somewhere as
   defined, or can be invoked in any situation parallelly. While other
   functions may only execute in certain cricumstances

