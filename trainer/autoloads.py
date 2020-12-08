from typing import List, Dict, Union, Iterable, Callable, Tuple
import torch
import abc


# TODO: Automatic separate code for parallel and non parallel
#       execution
class ModelStep(abc.ABC):
    """A ModelStep is an abstract class for data processing by the model.

    It's a class around a function by which inputs are sent through the model,
    outputs and losses are collected, loss is sent backward and other such tasks
    in a systematic manner. It provides a standard interface for sending and
    collecting data from the models with the :class:`Trainer`.

    A contrived example demonstrating the need and execution flow::

        class ExampleModelStep(ModelStep):
            def __call__(self, batch):
                # assuming self._model_names["foo"] is in self.models
                model_1 = self.models[self._model_names["foo"]]
                model_2 = self.models[self._model_names["bar"]]
                criterion_1 = self.criteria[self._criteria_names["foo"]]  # criterion for foo
                criterion_2 = self.criteria[self._criteria_names["bar"]]  # criterion for bar
                inputs, labels = batch
                inputs = model_1.to_(inputs)
                labels = model_1.to_(labels)
                if not self.test:
                    model_1.train()
                    model_1._optimizer.zero_grad()
                # NOTE: These should be checked for errors as order of execution may be important
                inter_vals = model_1(inputs)
                loss_1 = criterion_1(inter_vals, labels)
                final_vals = model_2(inter_vals, inputs)
                loss_2 = criterion_2(final_vals, inputs)  # maybe reconstruction loss
                if not self.test:
                    loss_1.backward()
                    loss_2.backward()
                # NOTE: values are model and criterion specific
                return {"losses": {"foo": loss_1.detach().item(), "bar": loss_2.detach().item()},
                        "outputs": {"foo": inter_vals.detach(), "bar": final_vals.detach()},
                        "labels": labels.detach(), "total": len(labels)}


        def check_foo(foo):
            try:
                foo(torch.randn(some_shape))
                return True
            except Exception:
                return False

        # etc.

    Assuming trainer.models is `{"Foo": Foo, "Bar": Bar, "OtherFoo": OtherFoo}`
    and in model_params the criteria are given as `{"foo": "ce_loss", "bar": "mse_loss"}`
    with critera as `{"ce_loss": torch.nn.CrossEntropyLoss, "mse_loss": torch.nn.MSELoss}`
    then::

        example_step = ExampleModelStep(model_names={"foo": "Foo", "bar": "Bar"},
                                        criteria_names={"foo": "ce_loss", "bar": "mse_loss"},
                                        checks={"foo": check_foo, "bar": check_bar})
        example_step.returns = {"losses", "outputs", "labels", "total"}

        # In trainer the models and criteria will always be thus:
        example_step.models = trainer.models
        example_step.criteria = trainer.criteria
        example_step.train = True  # set to train

        # Executed anywhere with a batch
        retval = example_step(batch)

        # later at test time
        example_step.test = True
        retval = example_step(batch)

        # much later, change only one model, checks performed automatically
        example_step.set_models({"foo": NewModelFoo})
        retval = example_step(batch)

    """

    def __init__(self, model_names: Union[Dict[str, str], Iterable[str]],
                 criteria_names: Dict[str, str],
                 checks: Dict[str, Callable], **kwargs):
        self._test = False
        self.models = None
        self.criteria = None
        if isinstance(model_names, dict):
            self._model_names = model_names
        else:
            self._model_names = {x: x for x in model_names}
        if not all(self._model_names[x] in criteria_names for x in self._model_names):
            raise AttributeError("Must have criterion for each model")
        self._criteria_names = criteria_names
        if not all(self._model_names[x] in checks for x in self._model_names):
            raise AttributeError("Must have checks for each model")
        self._checks = checks
        self._returns = None
        self._mode = None

    @abc.abstractmethod
    def __call__(self, batch: Iterable) -> Dict:
        """Call the Step

        Args:
            batch: A data specific iterable of values

        :meth:`__call__` is provided by the user and can have different modes.
        Standard modes are `train` and `test`.

        The execution flow and artefacts accumulated can depend on the
        modes. They have to be implemented by the user.

        """
        pass

    def set_models(self, models: Dict[str, str]) -> Dict[str, bool]:
        """Set the models which will be used.

        It's only a name mapping. `models` are handled by the trainer, but which
        model will be called is determined dynamically at run time. However
        because :attr:`checks` are `model` and `step` specific so they're
        checked here.

        Args:
            models: :class:`dict` of models which will be set

        ``models`` must be a :class:`dict` like ``{"internal_name": {"external_name": model}}``.

        """
        if not all(x in self._model_names for x in models):
            return False
        statuses = {}
        for key, val in models.items():
            status = self.check_model(key, self.models[val])
            if status:
                self._model_names[key] = val
        return statuses

    def set_checks(self, checks: Dict[str, Callable[[torch.nn.Module], bool]]):
        """Set the checks for the models and criteria.

        Args:
            checks: A :class:`dict` of {model_name: check_func} where check_func
                    is a function which takes a :class:`torch.nn.Module`
                    as input and returns an instance of :class:`bool`

        For example, one can verify that the output from each model is of a
        certain shape.  Criteria are more dynamic and are not checked here.

        """
        if not all(x in self._model_names for x in checks):
            raise ValueError("All model names should be in checks")
        self._checks = checks

    def check_model(self, model_name: str, model: torch.nn.Module) -> bool:
        return self._checks[model_name](model)

    @property
    def model_names(self) -> Dict[str, str]:
        """Return the model names."""
        return self._model_names

    @property
    def mode(self):
        """:attr:`mode` can be other than :attr:`test` and :attr:`train`"""
        if self._mode is not None:
            return self._mode
        else:
            return "train" if self._train else "test"

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def train(self):
        return not self._test

    @train.setter
    def train(self, x):
        self._test = not x

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, x):
        self._test = x

    @property
    def returns(self):
        """Names of artefacts returned by :meth:`__call__`.

        The step function can be different modes according to the modes
        supported. :attr:`returns` can be different for different modes.::

            self._returns["train"] == {"loss", "total"}
            self._returns["test"] == {"loss", "outputs", "label", "total"}
            self._returns["other"] == {"loss", "outputs", "label", "total", "other_metric"}

        "total" should always be returned by the step function and is implied.

        """
        return self._returns

    @returns.setter
    def returns(self, x: Iterable[str]):
        if "total" not in x:
            raise AttributeError("\"total\" must be present in returns")
        self._returns = x


class ClassificationStep(ModelStep):
    """Standard Classification training step.

    Arguments for parent class :class:`ModelStep`.

    Args:
        model_names: Name of the model
        criteria_names: Name of the model

    All `steps` are given `models`, `criteria`, `batch` as input. It's up to the
    `step` to determine how to use any or all of them.

    shapes and types of the batch should be handled by the data provider. This
    is just an example convenience wrapper on top of a forward model call. The
    return values and their format is the important part here.

    Loss is determined according to given criterion, between input and output
    values. Loss in classification is usually one of the `cross_entropy` losses
    from :mod:`torch.nn` or :mod:`torch.nn.functional`

    """
    def __call__(self, batch):
        """Call the Step. In this example models must have a key "net" which corresponds
        to a model which is compatible with this function.

        Args:
            batch: A data specific iterable of values

        """

        # "net" is the model name set at initialization
        # self._model_names["net"] refers to any other model that may
        # have been set later.
        net = self._model_names["net"]
        model = self.models[net]  # model initially named "net"
        criterion = self.criteria[self._criteria_names[net]]  # criterion for "net"
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
