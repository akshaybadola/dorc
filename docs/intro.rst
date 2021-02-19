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
trainer as the need arises and according to the commands given and availability.

Special directives can be given which would ensure automatic device
allocation.

- GPUs (or systems) themselves can be allocated to the trainer in
  :code:`trainer_params`.
- After which the models themselves can have which can set which devices they're
  active on. We use a :code:`_device` attr on the model to determine which device to
  assign to which model.
- Special device names like :code:`auto`, :code:`parallel` can indicate that the model is
  requested to be spread over all the possible GPUs (or systems). Otherwise
  device names.

1. If there are multiple models and gpus then the models are distributed
   among the gpus such that no more than one model is on each device.

   - If the number of gpus is greater than number of models then larger
     models are spread across multiple devices.
   - The aim would be to make sure that for a given batch of data, the
     forward call for each model is balanced.

2. If one model and multiple gpus are given, then the model is spread
   across all the models.
3. If the number of models is greater than the number of gpus then again
   the balancing principle applies, in this case however, the models may
   share devices.
4. If both "auto" or "parallel" are given for some models and explicit devices
   are given for some others, then those with devices given will be allocated
   first and the rest will be balanced later.

Models (including new model definitions) can be loaded or unloaded on
demand. This is a switching capability in case one wants to quickly compare two
models on the same trainer instance. See :ref:`models-root`


User Functions
--------------

1. :code:`user_func` shouldn't be given access to the trainer instance itself as
   it may cause unwanted states
2. :code:`hooks` may be specific functions which can be given access to specific
   locals of particular functions
3. :code:`user_func` is the most generic function and has arbitrary call and
   return values
4. In contrast :code:`hooks` would be more restricted and while adding or
   removing a hook certain checks can be performed
5. :code:`user_funcs` maybe invoked in the middle of the program somewhere as
   defined, or can be invoked in any situation parallelly. While other
   functions may only execute in certain cricumstances

