import os
import sys
import json
import logging


def _dump(x):
    return json.dumps(x, default=lambda o: f"<<{type(o).__qualname__}>>")
    # return json.dumps(x, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


def gen_file_logger(logdir, log_file_name):
    logger = logging.getLogger('default_logger')
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    # existing_files = [f for f in os.listdir(logdir) if f.startswith(log_file_name)]
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file):
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return log_file, logger


def get_backup_num(filedir, filename):
    backup_files = [x for x in os.listdir(filedir) if x.startswith(filename)]
    backup_maybe_nums = [b.split('.')[-1] for b in backup_files]
    backup_nums = [int(x) for x in backup_maybe_nums
                   if any([_ in x for _ in list(map(str, range(10)))])]
    if backup_nums:
        cur_backup_num = max(backup_nums) + 1
    else:
        cur_backup_num = 0
    return cur_backup_num


def gen_file_and_stream_logger(logdir, log_file_name, file_loglevel=None,
                               stream_loglevel=None):
    logger = logging.getLogger('default_logger')
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file):
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    if stream_loglevel.lower() == "info":
        stream_handler.setLevel(logging.INFO)
    elif stream_loglevel.lower() == "debug":
        stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    if file_loglevel.lower() == "info":
        file_handler.setLevel(logging.INFO)
    elif file_loglevel.lower() == "debug":
        file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return log_file, logger
