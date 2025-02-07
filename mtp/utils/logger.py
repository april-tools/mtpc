import logging

logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, master_process):
        self.master_process = master_process

    def __call__(self, message):
        if not self.master_process:
            return
        
        logger.info(message)
