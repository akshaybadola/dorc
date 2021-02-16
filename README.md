# Deep learning ORChestrator

DORC is a framework to manage training, validation, evaluation and dissection of
`Deep Learning` models over multiple machines having multiple GPUs. Its need
arose as a requirement to to monitor training over multiple machines during the
lockdown with uncertain network and power availability and over time more and
more features were added.

## Features
### Training and other functions
- One click remote training, validation with automatic logging and checkpoints.
- Pause and resume training/validation and run any alternative evaluation or
  dissection command.
- A unified abstraction of a `Model` which can easily be distributed over
  multiple GPUs and machines over a network.
- Run parallel additional functions without interrupting training.

### Ease of use
- Managemet console built with `ReactJS`.
- Live plotting and monitoring.

### Model abstraction and related features
- Modify/add/delete a Model during a training session with a simple
  pause/resume.
- Load/save weights to any given model.
- Multiple models with minor differences available in the same training session

### Device Management
- Automatic (`torch.Dataparallel`) or custom device allocation to each model.
- Device allocation over distributed machines.

### Utilities
- Remote file viewinig and editing.
- Portable JSON configuration of all training instances.

### Extensibility
- Extensibility via user uploaded functions in real time.
- View and modify code of any part of the model or weight updates.


See the documentation for more details.

## Usage
