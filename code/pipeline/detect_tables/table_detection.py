import logging
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
from pathlib import Path

class TableDetection:
    """
    Class for table detection using a transformer model.
    """

    def __init__(self, cache_folder, logger):
        """
        Initialise the table detection object.
        Sets device to cuda is GPU is available.
        If model files are not found in the cache folder, they are downloaded and saved for reuse.

        - cache_folder (str): Folder path to store model cache files
        - logger (Logger): Logger object
        """

        model = 'microsoft/table-transformer-detection'
        self.logger = logger

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.Log(f'{self.device} is available', logging.INFO)

        model_cache_folder = Path.joinpath(Path(cache_folder), Path(model.replace(r'/', '-')))
        try:
            self.table_detection_model = TableTransformerForObjectDetection.from_pretrained(model_cache_folder)
        except EnvironmentError:
            self.logger.Log("Downloading transformer model files!", logging.INFO)
            self.table_detection_model = TableTransformerForObjectDetection.from_pretrained(model, cache_dir=model_cache_folder)
            self.table_detection_model.save_pretrained(model_cache_folder)

        self.table_detection_model.to(self.device)

        self.image_processor = AutoImageProcessor.from_pretrained(model)
        self.image_processor.size['shortest_edge'] = 800

    def detect_tables(self, input_path, output_path, padding, threshold, save_temp_files, unix_timestamp):
        """
        Method to detect tables in the image by calling the transformer model on preprocessed image.
        
        - input_path (str): Input image path
        - output_path (str): Folder path to store all files generated at output
        - padding (int): Padding to crop detected tables
        - threshold (float): Threshold for table detection
        - save_temp_files (bool): Bool - set true to save temp files (cropped table image, box cordinates)
        - unix_timestamp (int): Make every file name unique using timestamp
        
        * returns tables (List): List of PIL images of cropped tables
        """
        
        image = Image.open(input_path).convert("RGB")

        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs.to(self.device)

        outputs = self.table_detection_model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        if save_temp_files:
            temp_output_folder = Path.joinpath(Path(output_path), Path('temp'))
            temp_output_folder.mkdir(parents=True, exist_ok=True)

        c = 0
        tables = []

        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if self.table_detection_model.config.id2label[label.item()] == 'table':
                table_name = f'{Path(input_path).stem}_table_{c}'
                table = image.crop([box[0].item()-padding, box[1].item()-padding, box[2].item()+padding, box[3].item()+padding])
                
                if save_temp_files:
                    table_image_path = Path.joinpath(temp_output_folder, Path(f'{table_name}_{unix_timestamp}.png'))
                    table.save(table_image_path)

                    f = open(Path.joinpath(temp_output_folder, Path(f"{table_name}_box_{unix_timestamp}.txt")), "w")
                    f.write(f"Score: {score}\nBox: {box.tolist()}")
                    f.close()

                c += 1
                tables.append(table)

                self.logger.Log(f'{table_name} --> Score: {score}, Box: {box.tolist()}', logging.INFO)

        return tables