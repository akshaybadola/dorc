.. trainer documentation master file, created by
   sphinx-quickstart on Sun Jan  5 16:51:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to DORC's documentation!
===================================

DORC stands for Deep learning ORchestrator and is envisaged to manage training,
validation, evaluation and dissection of `Deep Learning` models over multiple
machines.

It has features like:

- Manage multiple training sessions.
- Spread the model over multiple GPUs (distributed training over the network is to be added).
- Pause and Resume any of those sessions independently.
- Run parallel additional functions without interrupting training.
- Add/Delete models after pausing the training session and resume
- Load/unload weights to any given model
- Upload/delete any arbitrary module or user defined function in the code
  environment without stopping training.
- Execute any user given function (on any subset of given data or otherwise) in
  parallel to training.
- Switch devices of models during training.
- View logs related to training from remote sessions.
- View remote files in a code editor and save it back to the machine.
- And many more...

You can browse the :ref:`source/contents:contents` or, get started with :ref:`dorc-root`

..
  .. _index-need:
  Need
  ----

  Which :ref:`index:Need`, :ref:`dorc-root` do we link to?

  Test, :ref:`source/dorc:Device Allocation`, :ref:`source/dorc:Need`, :mod:`torch.optim`

.. toctree::
   :hidden:
   :maxdepth: 2

   source/contents
   source/dorc
   source/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
