import torch


# TODO: This has to better defined with the execution logic
#       being inferred from the variable `execution_logic`
# TODO: to_string also has to be better implemented so that
#       most of the function is exported automatically and
#       only the relevant parts need to be filled in.
# TODO: Automatic separate code for parallel and non parallel
#       execution
# TODO: Currently there's no way of specifying which optimizer
#       to use. That has to be incorporated. Not sure how to
#       proceed.
class ModelStep:
    def __init__(self, model_names, execution_logic):
        self._model_names = model_names
        self._execution_logic = execution_logic

    def __call__(self, trainer, batch):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError


class ClassificationTrainStep:
    def __init__(self, model_name):
        self.expects = ["models.model_name", "optimizers.Adam", "criteria.criterion_ce_loss"]
        self.returns = [("metric", "cross_entropy_loss"), ("io", "outputs"), ("io", "labels"),
                        ("var", "total")]
        self._model_name = model_name

    # DONE: Apply data parallel here
    def __call__(self, wrp, batch):
        wrp.optimizers["Adam"].zero_grad()  # This code is faulty
        wrp.models[self._model_name].train()
        if wrp.device == "parallel":
            inputs, labels = batch[0].cuda(), batch[1].cuda()
        else:
            inputs, labels = batch[0].to(wrp.device), batch[1].to(wrp.device)
        outputs = wrp.models[self._model_name](inputs)
        loss = wrp.criteria["cross_entropy_loss"](outputs, labels)
        loss.backward()
        wrp.optimizers["default"].step()
        return {"cross_entropy_loss": loss.data.item().detach(), "outputs": outputs.detach(),
                "labels": labels.detach(), "total": len(labels)}

    def export(self):
        pass


class ClassificationTestStep:
    def __init__(self, model_name):
        self.expects = ["models.model_name"]
        self.returns = [("metric", "cross_entropy_loss"), ("io", "outputs"), ("io", "labels"),
                        ("var", "total")]
        self._model_name = model_name

    # DONE: Apply data parallel here maybe
    def __call__(self, wrp, batch):
        with torch.no_grad():
            wrp._set_models_eval()
            if wrp.device == "parallel":
                inputs, labels = batch[0].cuda(), batch[1].cuda()
            else:
                inputs, labels = batch[0].to(wrp.device), batch[1].to(wrp.device)
            outputs = wrp.models[self._model_name](inputs)
            loss = wrp.criteria["cross_entropy_loss"](outputs, labels)
        return {"cross_entropy_loss": loss.data.item().detach(), "outputs": outputs.detach(),
                "labels": labels, "total": len(labels)}

    def export(self):
        pass


# Example of using extra metrics
# extra_metrics = {"train": {"accuracy": {"function": accuracy,
#                                         "inputs": ["outputs", "batch[1]"]}},
#                  "val": {"accuracy": {"function": accuracy,
#                                       "inputs": ["outputs", "batch[1]"]}}}
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return float(correct)/len(predicted)
