import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pandas import read_csv
from re import findall
from io import StringIO

def load_image(image):
    """
    Functions for preprocessing image for LLM.
    REF: https://huggingface.co/OpenGVLab/InternVL2-8B#inference-with-transformers

    - image (PIL): PIL image being passed to the LLM.

    * returns pixel_values (tensor): Pixel values of the processed image.
    """

    transform = _build_transform()
    images = _dynamic_preprocess(image)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def _build_transform():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    input_size = 448
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def _dynamic_preprocess(image):
    min_num=1
    max_num=12
    image_size=448
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    return processed_images

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def postprocess_response(markdown_string):
    """
    Extract the dataframes from the LLM response.
    
    - markdown_string (str): The string response from the LLM.

    * returns tables (List): List of tables as dataframe.
    """
    
    # Split the markdown string into lines
    lines = markdown_string.strip().splitlines()

    tables = []
    table_lines = []
    in_table = False

    for line in lines:
        if '|' in line:
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        elif in_table:
            table = _parse_tables(table_lines)
            tables.append((table))
            in_table = False

    # Handle the last table if it wasn't followed by non-table text
    if in_table and table_lines:
        table = _parse_tables(table_lines)
        tables.append((table))
        
    return tables

def _parse_tables(table_lines):
    table_lines_corrected=[]
    num_colls = 0
    for line in table_lines: num_colls = max(num_colls, len(line.split('|'))-2)
    for line in table_lines:
        if len(line.split('|'))-2 < num_colls:
            line = line + ' |' * (num_colls - (len(line.split('|'))-2))
        table_lines_corrected.append(line)

    column_headings_line = [i for i in range(len(table_lines_corrected)) if len(findall('^[| -]*$', table_lines[i])) > 0]
    header = None
    if column_headings_line:
        header=0
        table_string = "\n".join(table_lines_corrected).strip('|')
    else:
        table_string = "\n".join(table_lines_corrected)

    table = read_csv(StringIO(table_string), sep="|", engine='python', skipinitialspace=True, header=header)
    table.dropna(axis=1, how='all', inplace=True)
    table.fillna('', inplace=True)
    table.reset_index(drop=True, inplace=True)
    
    return table