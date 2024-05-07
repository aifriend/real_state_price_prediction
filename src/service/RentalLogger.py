# Create a logger
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_directory = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(project_directory))

logger = logging.getLogger('rLogger')
logger.setLevel(logging.INFO)

# Create a file handler
fh = logging.FileHandler(Path.cwd().joinpath('rental.log'))
fh.setLevel(logging.INFO)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
