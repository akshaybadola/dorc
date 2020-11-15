import sys
from trainer import FlaskInterface


if __name__ == '__main__':
    hostname = sys.argv[1]
    port = int(sys.argv[2])
    data_dir = sys.argv[3]
    if len(sys.argv) > 4 and sys.argv[4].lower() == "false":
        production = False
    else:
        production = True
    sys.argv = [sys.argv[0]]
    sys.path.append(data_dir + "/../../")
    FlaskInterface(hostname, port, data_dir, production=production)
