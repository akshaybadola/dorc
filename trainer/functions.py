import torch


class ClassificationTrainStep:
    def __init__(self):
        self.returns = [("metric", "cross_entropy_loss"), ("io", "outputs"), ("io", "labels"),
                        ("var", "total")]

    # DONE: Apply data parallel here
    def _train_step(self, wrp, batch, model_name):
        wrp.optimizers["default"].zero_grad()
        wrp.models[model_name].train()
        if wrp.device == "parallel":
            inputs, labels = batch[0].cuda(), batch[1].cuda()
        else:
            inputs, labels = batch[0].to(wrp.device), batch[1].to(wrp.device)
        outputs = wrp.models[model_name](inputs)
        loss = wrp.criteria["cross_entropy_loss"](outputs, labels)
        loss.backward()
        wrp.optimizers["default"].step()
        return {"cross_entropy_loss": loss.data.item().detach(), "outputs": outputs.detach(),
                "labels": labels.detach(), "total": len(labels)}


class ClassificationTestStep:
    def __init__(self):
        self.returns = [("metric", "cross_entropy_loss"), ("io", "outputs"), ("io", "labels"),
                        ("var", "total")]

    # DONE: Apply data parallel here maybe
    def __call__(self, wrp, batch):
        with torch.no_grad():
            wrp._set_models_eval()
            if wrp.device == "parallel":
                inputs, labels = batch[0].cuda(), batch[1].cuda()
            else:
                inputs, labels = batch[0].to(wrp.device), batch[1].to(wrp.device)
            outputs = wrp.model(inputs)
            loss = wrp.criteria["cross_entropy_loss"](outputs, labels)
        return {"cross_entropy_loss": loss.data.item().detach(), "outputs": outputs.detach(),
                "labels": labels, "total": len(labels)}


# Example of using extra metrics
# extra_metrics = {"train": {"accuracy": {"function": accuracy,
#                                         "inputs": ["outputs", "batch[1]"]}},
#                  "val": {"accuracy": {"function": accuracy,
#                                       "inputs": ["outputs", "batch[1]"]}}}
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return float(correct)/len(predicted)
