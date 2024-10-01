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
        Method to extract the table data. Table detection, content recognition steps happen here.

        * returns tables (List): List of tables.
        """

        table_detection = TableDetection(self.config['cache'], self.logger)
        detected_tables = table_detection.detect_tables(self.config['input_path'], self.config['output_path'], self.config['padding'], self.config['save_temp_files'], int(self.start_time))

        if not detected_tables:
            self.logger.Log('No tables found!', logging.INFO)
            self.logger.Log(f'Pipeline completed for {self.config["input_path"]}', logging.INFO)
            return []

        content_extraction = ContentExtraction(self.config['max_new_tokens'], self.config['cache'], self.config['load_in_8bit'], self.logger)
        extracted_tables_page = [content_extraction.extract_content(table, self.config['use_pipeline_a']) for table in detected_tables]

        tables = [
                        extracted_table
                        for extracted_tables_image in extracted_tables_page
                        for extracted_table in extracted_tables_image
                    ]
        
        pipeline_name = "pipeline_a" if self.config['use_pipeline_a'] else "pipeline_b"
        for i, table in enumerate(tables):
            content_extraction.save_table(table, Path(self.config['output_path']), Path(self.config['input_path']).stem, pipeline_name, i, self.start_time)

        self.logger.Log(f'Pipeline completed for {self.config["input_path"]}', logging.INFO)
        self.logger.Log('Total time taken = {:.2f} seconds'.format(time()-self.start_time), logging.INFO)

        return tables