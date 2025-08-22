import logging
import os

def setup_logging(log_file='heart_disease_analysis.log', level=logging.INFO):
    """
    Sets up logging to both console and a file.
    """
    # Create logger
    logger = logging.getLogger('heart_disease_analysis')
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler() # Console handler
    f_handler = logging.FileHandler(log_file) # File handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    # Ensure handlers are not duplicated if setup_logging is called multiple times
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
