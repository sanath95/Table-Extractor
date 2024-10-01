from pipeline import TableExtractorPipeline

if __name__ == '__main__':
    """
    The main file that is called by the user.
    """

    pipeline = TableExtractorPipeline('./code/config.json')
    
    try:
        fstrings = pipeline.extract_tables()
    except Exception as e:
        import logging
        logging.error(f'An error occurred: {e}')