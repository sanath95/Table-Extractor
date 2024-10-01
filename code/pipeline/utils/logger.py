import logging

class Logger:
    """
    Class for logging. Supported logging levels - info, warning, and error.
    """

    def __init__(self, file_path=None):
        """
        Initialise the logger with a basic config.
        
        - file_path: Path where the log file should be stored. If None, logs are printed on the console.
        """

        logging.basicConfig(
            filename=file_path,
            encoding="utf-8",
            filemode="a",
            format="{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%d-%m-%Y %H:%M",
        )
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
    
    def Log(self, text, level):
        """
        Log the event.
        
        - text (str): Message to be logged.
        - level (const): Log level.
        """
        
        match level:
            case logging.INFO:
                self.logger.info(text)
            case logging.WARNING:
                self.logger.warning(text)
            case logging.ERROR:
                self.logger.error(text)
            case _:
                self.logger.warning('Invalid logging level!')
                self.logger.warning(text)