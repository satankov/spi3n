import os
import logging


def create_dir(path):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)


def set_logger(
        filename,
        path='',
        level=20):
    return \
        logging.basicConfig(
            filename=f'{path}{filename}',
            level=level,
            format='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s'
        )
