import logging
from pathlib import Path
from time import time

from pipeline.detect_tables import TableDetection
from pipeline.extract_content import ContentExtraction
from pipeline.utils import ConfigParser, Logger

class TableExtractorPipeline:
    """
    Class for table extraction pipeline - the heart of the application.
    """

    def __init__(self, config_path):
        """
        Initialise the pipeline.
        Get configuration parameters.
        Initialise the logger.
        
        - config_path (string): Path to config file
        """
        self.config = ConfigParser(config_path).get_config()
        self.logger = Logger(self.config['log_file_path'])
        self.logger.Log('Pipeline started', logging.INFO)
        self.start_time = time()

        for k, v in self.config.items(): self.logger.Log(f'CONFIG ** {k}: {v}', logging.INFO)
        
    def extract_tables(self):
        """
        Method to extract the table data as formatted strings. Table detection, content recognition steps happen here.

        * returns fstrings (List): List of fstrings extracted.
        """

        table_detection = TableDetection(self.config['cache'], self.logger)
        detected_tables = table_detection.detect_tables(self.config['image_path'], self.config['output_path'], self.config['padding'], self.config['save_temp_files'], int(self.start_time))

        if not detected_tables:
            self.logger.Log('No tables found!', logging.INFO)
            self.logger.Log(f'Pipeline completed for {self.config['image_path']}', logging.INFO)
            return []

        llm = ContentExtraction(self.config['max_new_tokens'], self.config['cache'], self.config['load_in_8bit'], self.logger)
        fstrings_page = [llm.extract_fstring(table, self.config['use_pipeline_a']) for table in detected_tables]

        fstrings = [
                        fstrings_table
                        for fstrings_image in fstrings_page
                        for fstrings_table in fstrings_image
                    ]

        pipeline_name = "pipeline_a" if self.config['use_pipeline_a'] else "pipeline_b"
        fstrings_output_path = Path.joinpath(Path(self.config['output_path']), Path(f'output_{pipeline_name}_{int(self.start_time)}.json'))
        llm.save_fstrings(fstrings, fstrings_output_path, Path(self.config['image_path']).stem)

        self.logger.Log(f'Pipeline completed for {self.config['image_path']}', logging.INFO)
        self.logger.Log('Total time taken = {:.2f} seconds'.format(time()-self.start_time), logging.INFO)

        return fstrings