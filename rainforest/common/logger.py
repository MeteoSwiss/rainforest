import logging
import gzip
import os
import sys
from datetime import datetime


class DailyGzipFileHandler(logging.Handler):
    def __init__(self, base_dir="/srn/las/log/", log_basename="run_rfo_rt_mode"):
        super().__init__()
        self.base_dir = base_dir
        self.current_day = None
        self.gzfile = None
        self._open_new_file(log_basename)

    def _open_new_file(self, basename="run_rfo_rt_mode"):
        """Open a new gzip log file for the current day."""
        if self.gzfile:
            self.gzfile.close()

        self.current_day = datetime.now().strftime("%d")
        current_date = datetime.now().strftime("%Y%m%d")
        logfilename = f"{basename}_{current_date}.log.gz"
        log_dir = os.path.join(self.base_dir, self.current_day)
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, logfilename)
        if os.path.exists(log_path):
            os.remove(log_path)  # overwrite last month’s file

        self.gzfile = gzip.open(log_path, "at", encoding="utf-8")

    def emit(self, record):
        """Write log record to gzip file, rotating if needed."""
        today = datetime.now().strftime("%d")
        if today != self.current_day:
            # New day — open a new file
            self._open_new_file()

        msg = self.format(record)
        self.gzfile.write(msg + "\n")
        self.gzfile.flush()

    def close(self):
        if self.gzfile:
            self.gzfile.close()
        super().close()


def get_qpe_rt_logger(log_dir="/srn/las/log/", verbose=False):
    """
    Creates the 'Rainforest RT' logger and attaches its handler to the root logger
    so that all loggers in the app propagate to the same file.
    """
    logger = logging.getLogger("Rainforest RT")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create and configure handler
    handler = DailyGzipFileHandler(log_dir)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Attach handler to the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    return logger


def get_logger(name=None, log_dir=None, verbose=False):
    """
    Returns a logger that shares the root logger's handlers.
    If log_dir is given, logs go to a file; otherwise, logs go to stdout.
    """
    if not name:
        name = __name__

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers only once to avoid duplicates
    if not root_logger.handlers:
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            logfile = os.path.join(log_dir, f"{name}.log")
            handler = logging.FileHandler(logfile)
        else:
            handler = logging.StreamHandler(sys.stdout)

        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        root_logger.addHandler(handler)

    return logger
