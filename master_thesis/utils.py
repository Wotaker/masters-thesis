import logging

LOGLEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "PLOT": 15,  # Add custom level for plotting
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}
logging.addLevelName(LOGLEVEL_MAP["PLOT"], "PLOT")