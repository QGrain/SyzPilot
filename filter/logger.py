import datetime
import logging
import os
from pathlib import Path

class TrainLogger:
    def __init__(self, log_root: str="./logs/", log_name: str="TrainLogger") -> None:
        _now = datetime.datetime.now()
        _filename = _now.strftime("%Y%m%d_%H%M%S")
        _dir = Path(log_root)
        _dir.mkdir(exist_ok=True)

        self.path = os.path.join(log_root, f"{_filename}-{log_name}.log")
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        fh = logging.FileHandler(self.path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(f'Timestamp: {_now.strftime("%Y-%m-%d  %H:%M:%S:%f")}')
        self.logger.info(f'Timestamp: {_now.strftime("%A, %B %-d, %Y   %-I:%M:%S %p")}')

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)


# logger
if __name__ == "__main__":
    log = TrainLogger()
    log.debug("DEBUG")
    log.info("INFO")
