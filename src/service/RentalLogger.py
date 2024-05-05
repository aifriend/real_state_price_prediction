# Create a logger
import logging
from pathlib import Path

logger = logging.getLogger('rLogger')
logger.setLevel(logging.INFO)

# Create a file handler
fh = logging.FileHandler(
    Path.cwd().joinpath('../', 'rental.log'))
fh.setLevel(logging.INFO)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
