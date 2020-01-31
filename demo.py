from trainer.trainer import Trainer
from trainer import interfaces
from tests.setup import config

config["uid"] = "demo_trainer"
trainer = Trainer(**config)
flask = interfaces.FlaskInterface("0.0.0.0", 30000, trainer, True)
flask.start()
