from trainer.trainer import Trainer
from trainer import interfaces
from tests.setup import config

config["uid"] = "demo_trainer"
trainer = Trainer(**config)
trainer._init_all()
flask = interfaces.FlaskInterface("0.0.0.0", 20202, trainer, True)
flask.start()
