import logging


class TraderLogger(object):
    def __init__(self, level):
        logging.basicConfig(
            filename='logs/exec-log.txt',
            filemode='a',
            format='%(asctime)s | %(message)s', level=level)

    def write_info(self, mess):
        logging.info(mess)


