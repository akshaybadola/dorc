import os
from trainer.daemon import _start_daemon


daemon = _start_daemon("0.0.0.0", 20202, ".demo_dir", os.path.abspath("dist"),
                       os.path.abspath("dist"), os.path.abspath("."))
