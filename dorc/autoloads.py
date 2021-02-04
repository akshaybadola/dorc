import torch
from dorc.trainer.model import ModelStep


class ClassificationStep(ModelStep):
    """Standard Classification training step.

    Arguments for parent class :class:`ModelStep`.

    In Addition to the args for :class:`ModelStep` any number of variable
    `*args` or `**kwargs` can be passed to faciliate the working of the step.

    We can pass a "val" attribute to the existing :class:`ModelStep`, which can
    change the `__call__` accordingly

    Example::
        class ClassificationStepWithVal(ModelStep):
            def __init__(self, *args, **kwargs):
                # Other code
                self._val = kwargs["val"]

            # Other code

            # The `mode` can thus be changed like this
            @property
            def mode(self):
                if self._val and not self.train:
                    return "val"
                elif self.train:
                    return "train"
                else:
                    return "test"

            @mode.setter
            def mode(self, x):
                if x == "train":
                    self.train = True
                elif x == "val":
                    self.train = False
                    self._val = True
                elif x == "test":
                    self.train = False
                    self._val = False
                else:
                    raise ValueError(f"Invalid value of mode {x}")

            def __call__(self, batch):
                # Other lines of code

                if self.mode == "val":
                    # do something
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._returns = ["loss", "outputs", "labels", "total"]
        self._logs = ["loss"]

    def __call__(self, batch):
        """Call the Step. In this example models must have a key "net" which corresponds
        to a model which is compatible with this function.

        Args:
            batch: A data specific iterable of values

        """
        model = self.models["net"]
        criterion = self.criteria["net"]  # criterion for "net"
        inputs, labels = batch
        inputs = model.to_(inputs)
        labels = model.to_(labels)
        if self.train:
            model.train()
            model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if self.train:
            loss.backward()
        return {"loss": loss.detach().item(), "outputs": outputs.detach(),
                "labels": labels.detach(), "total": len(labels)}


# Example of using extra metrics
# extra_metrics = {"train": {"accuracy": {"function": accuracy,
#                                         "inputs": ["outputs", "batch[1]"]}},
#                  "val": {"accuracy": {"function": accuracy,
#                                       "inputs": ["outputs", "batch[1]"]}}}
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return float(correct)/len(predicted)


# NOTE: Why are these functions doing this requires provides anyway? Is this
#       type checking or error checking? If I want robust adherence to a spec, I
#       should make provides and requires abstract functions. Checkable has
#       "virtual" functions/properties "provides" and "requires" like below. It
#       is to facilitate module interaction. Two components can easily and
#       immediately know if they can work together or not, instead of making
#       assumptions, forcing things and creating subtle difficulties later.
#
#       On the theoretical front, generally functions would be defined by the
#       signatures, leaving the implementation to the programmer. However, I
#       would like to have a deeper understanding of the components so that
#       these things aren't just defined by the signature but also their
#       behaviour. Can static typing help here?
#
#       Adhering to an Interface solves these problems, but what if the
#       interface is needed to be flexible? In the below example, metrics needs
#       to be a list, but there's an additional attribute that needs to be there
#       which is when the function is to be called. Should I make it a property?
#       @property
#       def when(self):
#           return "train_end"
#
#       In CheckAccuracy below the data structure of metrics is implicit, while
#       it actually is checked at CheckFunc. Perhaps a better check can be put
#       there. Except we won't know the type and structure of "metrics" until
#       it's called, which may leak an error later in the code. Perhaps mypy or
#       pyright can help here.  Perhaps I should use type annotations for the
#       functions. They seem like a good idea, especially since they can
#       describe complicated types. To facilitate communication over the network
#       they should be json-serializable also.
class CheckFunc:
    def __init__(self, when):
        """Example metrics:

        {'train': {'loss': {0: 7.294781831594614}, 'samples': {0: 6561}, 'perplexity': {}},
        'val': {'loss': {}, 'samples': {0: 0}, 'perplexity': {}, 'sentence_metrics': {}},
        'test': {'loss': {}, 'samples': {0: 0}, 'perplexity': {}, 'sentence_metrics': {}}}

        :returns: None
        :rtype: None

        """
        assert when in ["train", "val", "test"]
        self._requires = {}
        self._provides = {}

    @property
    def requires(self):
        if not isinstance(self._requires["metrics"], list):
            metrics = self._requires.pop("metrics")
            assert isinstance(metrics, str)
            self._requires["metrics"] = [metrics]
        return self._requires

    @property
    def provides(self):
        return self._provides

    def __call__(self, metrics) -> bool:
        return False


class CheckGreater(CheckFunc):
    def __init__(self, when):
        super().__init__(when)

    def __call__(self, metrics):
        raise NotImplementedError


class CheckGreaterName(CheckGreater):
    def __init__(self, when, name):
        super().__init__(when)
        self._name = name
        self._requires = {"when": when, "metrics": name}

    def __call__(self, metrics):
        vals = [*metrics[self._name].items()]
        if vals:
            vals.sort(key=lambda x: x[0])
            vals = [v[1] for v in vals]
            if all(vals[-1] > x for x in vals[:-1]):
                return True
            else:
                return False
        else:
            return False


class CheckLesserName(CheckGreater):
    def __init__(self, when, name):
        super().__init__(when)
        self._name = name
        self._requires = {"when": when, "metrics": name}

    def __call__(self, metrics):
        vals = [*metrics[self._name].items()]
        if vals:
            vals.sort(key=lambda x: x[0])
            vals = [v[1] for v in vals]
            if all(vals[-1] < x for x in vals[:-1]):
                return True
            else:
                return False
        else:
            return False


class CheckAccuracy(CheckGreaterName):
    def __init__(self, when):
        super().__init__(when, "accuracy")
