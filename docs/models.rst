.. _models-root:

Model
=====

A `Model` is a function over which you'd iterate during training until your
desired objective is achieved. It'll have some :code:`inputs` and :code:`outputs`


Models Lifecycle
----------------

Models are given as :code:`class` definitions and they can be initialized or
dumped by the trainer according to the need.

They work in conjunction with a :ref:`modelstep-root` which is an abstraction of
a function which takes batch :code:`inputs` and returns the :code:`outputs` and
:code:`losses` and other artefacts.

Models can be loaded or unloaded on demand by the user:


Model Loading
-------------

A model is said to be :code:`loaded` if it's
a. Assigned to the :ref:`modelstep-root`
b. And is loaded into the specified device memory, and hence is ready to accept
inputs.

Being assigned to a :class:`~dorc.trainer.model.ModelStep` automatically loads a
model into the device memory.

Multiple Models can exist in the same trainer within a training
:doc:`/session`. If the :ref:`modelstep-root` requires more than one models,
they are loaded as they're assigned to the :ref:`modelstep-root`.

To check the model names and currently allocated model call :code:`/props/active_models`.
