from prismWriter import * #moved to its own package for reuse in other projects
import logging
logging.getLogger('prismWriter').addHandler(logging.NullHandler())
#also warn the user about deprecated location
logger = logging.getLogger(__name__)
logger.warning("The prism_writer_gui module has been moved to the prismWriter package. Please update your imports accordingly.")