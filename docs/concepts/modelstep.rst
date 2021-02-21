.. _concepts-modelstep:

Model Step
==========

1. Takes :code:`inputs` as a batch.
2. Places it on appropriate devices.
3. Calls the :code:`model` (or :code:`models`) on the :code:`inputs`.
4. Gathers the :code:`outputs`, and calculates the losses with the given :code:`criteria` (See :ref:`criteria-root`)
5. Calls the :code:`optimizer` if required.
6. And returns the :code:`losses` and any other artefacts generated during the process.


For the implementation see :class:`~dorc.trainer.models.ModelStep`

