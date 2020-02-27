import sys
from trainer.interfaces import FlaskInterface


if __name__ == '__main__':
    hostname = sys.argv[1]
    port = int(sys.argv[2])
    data_dir = sys.argv[3]
    print("IN MAIN?")
    FlaskInterface(hostname, port, data_dir)
