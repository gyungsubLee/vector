import pandas as pd
import os
from tqdm import tqdm
import warnings
import numpy as np
import torch
from ultralytics.utils.metrics import mask_iou
import itertools
import matplotlib.image as mpimg

# Constants
WARNING_FILTER_ACTION = "ignore"
DEFAULT_IMAGE_BASE_PATH = "./data/imaterialist-fashion-2020-fgvc7/train"
TRAIN_CSV_PATH = "./data/imaterialist-fashion-2020-fgvc7/train.csv"
OUTPUT_FILE_PATH = "./data/outputs.json"
IMAGE_EXTENSION = ".jpg"
MASK_RESHAPE_ORDER = "F"
MASK_DTYPE = torch.float32
DEFAULT_OVERLAP_THRESHOLD = 0.9
NO_OVERLAP_IOU = 0
PAIR_COMBINATION_SIZE = 2
ATTRIBUTE_SEPARATOR = ","
EMPTY_ATTRIBUTE = ""
BATCH_IMAGE_WINDOW = 3
DEFAULT_BATCH_SIZE = 20
MAIN_BATCH_SIZE = 5
OUTPUT_JSON_ORIENT = "records"
FILE_WRITE_MODE_APPEND = "a"
FILE_WRITE_MODE_TRUNCATE = "w"
OUTPUT_LINE_BREAK = "\n"
TQDM_DESC = "Processing Batches"
START_MESSAGE = "Start process"

# Ignore warnings
warnings.filterwarnings(WARNING_FILTER_ACTION)

def create_separate_masks(annoations, class_ids, height, width):
    masks = []

    for segment, (pixel_str, class_id) in enumerate(zip(annoations, class_ids)):
        mask = np.zeros((height, width)).reshape(-1)
        splitted_pixels = list(map(int, pixel_str.split()))
        pixel_starts = splitted_pixels[::2]
        run_lengths = splitted_pixels[1::2]
        assert max(pixel_starts) < mask.shape[0]
        for pixel_start, run_length in zip(pixel_starts, run_lengths):
            pixel_start = int(pixel_start) - 1
            run_length = int(run_length)
            mask[pixel_start:pixel_start+run_length] = 1
        masks.append(mask.reshape((height, width), order=MASK_RESHAPE_ORDER))
    return masks

def flatten_mask(mask):

    flattened_mask = mask.flatten()
    mask_tensor = np.reshape(flattened_mask, (1, -1))

    mask_tensor = torch.tensor(mask_tensor, dtype=MASK_DTYPE)
    return mask_tensor

def check_overlap(mask1, mask2, threshold=DEFAULT_OVERLAP_THRESHOLD):
    """
    Determine if the overlap between two masks covers more than `threshold` of the smaller mask.
    """
    # Calculate IoU using the mask_iou function
    iou = mask_iou(mask1, mask2).item()
    if iou == NO_OVERLAP_IOU:
        return False
    
    # Calculate the areas of the masks
    area1 = mask1.sum().item()
    area2 = mask2.sum().item()
    
    # Determine the smaller mask
    smaller_area = min(area1, area2)
    
    # Calculate the intersection area based on IoU and union
    intersection = iou * (area1 + area2) / (1 + iou)
    
    # Check if the intersection covers more than threshold of the smaller mask
    if intersection / smaller_area > threshold:
        return True
    else:
        return False
    
def search_attribute_pairs(tmp_df, image_base_path=DEFAULT_IMAGE_BASE_PATH):
    tmp = tmp_df.reset_index(drop=True).copy()
    image = mpimg.imread(
        os.path.join(image_base_path, tmp.ImageId.unique()[0] + IMAGE_EXTENSION)
    )
    # binary mask 생성
    masks = create_separate_masks(tmp['EncodedPixels'], tmp['ClassId'], tmp['Height'].values[0], tmp['Width'].values[0])

    combinations = list(itertools.combinations(range(len(masks)), PAIR_COMBINATION_SIZE))

    pairs = list()

    # 모든 combination 고려
    for comb in combinations:
        # binary mask 선택
        mask1 = masks[comb[0]]
        mask2 = masks[comb[1]]
        # 비교를 위해 flatten
        flat1 = flatten_mask(mask1)
        flat2 = flatten_mask(mask2)
        # 두 binary mask들 중 작은 mask가 큰 mask와 90% 이상 픽셀을 공유하는지 여부 체크
        if check_overlap(flat1, flat2):
            # 둘 중 큰 mask를 선별하여 대표 mask로 설정
            if mask1.sum() > mask2.sum():
                pairs.append([comb[0], comb])
            else:
                pairs.append([comb[1], comb])
    return pairs

def merge_attribute_pairs(tmp_df, pairs):
    tmp = tmp_df.reset_index(drop=True).copy()
    # attribute이 없는 경우도 존재하기 때문에, string 값으로 변환
    tmp.loc[tmp['AttributesIds'].isna(), 'AttributesIds'] = EMPTY_ATTRIBUTE
    main_pairs = list(set([i[0] for i in pairs]))

    for mp in main_pairs:
        # 상위 카테고리가 포함된 pair 선택
        pair = [i[1] for i in pairs if i[0] == mp]
        # 상위 카테고리를 제외한 다른 id만 선택 == 하위 카테고리
        flat_pair = list(set([element for tuple_ in pair for element in tuple_]))
        sub_category = [i for i in flat_pair if i != mp]
        # 하위 카테고리들의 attribute를 하나로 병합
        sub_attributes = tmp.loc[sub_category, 'AttributesIds'].values
        sub_attributes = list(
            set(ATTRIBUTE_SEPARATOR.join(sub_attributes).split(ATTRIBUTE_SEPARATOR))
        )
        sub_attributes = ATTRIBUTE_SEPARATOR.join(sub_attributes)
        # 상위 카테고리의 second attribute로 저장
        tmp.loc[mp, 'second_AttributesIds'] = sub_attributes

    return tmp

### functions for batch processing
def divide_into_batches(dataframe, batch_size):
    """Yield successive n-sized chunks from dataframe."""
    images = dataframe['ImageId'].unique()
    batches = [images[i:i + BATCH_IMAGE_WINDOW] for i in range(0, len(images), batch_size)]

    for batch in batches:
        yield dataframe.loc[dataframe['ImageId'].isin(batch)]


def process_and_append_to_file(batch, image_base_path, output_file):
    processed_batch = [process_per_image((image, batch, image_base_path)) for image in batch['ImageId'].unique()]
    # Convert each DataFrame in processed_batch to a line-delimited JSON string
    with open(output_file, FILE_WRITE_MODE_APPEND) as f_out:
        for df in processed_batch:
            json_string = df.to_json(orient=OUTPUT_JSON_ORIENT)
            f_out.write(json_string + OUTPUT_LINE_BREAK)  # Write each batch as a new line

def process_per_image(args):
    image, anno, image_base_path = args

    # Filter the DataFrame for the current image
    tmp_df = anno[anno['ImageId'] == image].copy()
    # Search for attribute pairs within this image's data
    pairs = search_attribute_pairs(tmp_df, image_base_path)
    # If pairs are found, merge their attributes
    if len(pairs) > 0:
        tmp_df = merge_attribute_pairs(tmp_df, pairs)
    return tmp_df

def parallel_process_images(anno, image_base_path, output_file, batch_size=DEFAULT_BATCH_SIZE):
    batches = list(divide_into_batches(anno, batch_size))
    for batch in tqdm(batches, desc=TQDM_DESC):
        process_and_append_to_file(batch, image_base_path, output_file)

if __name__ == "__main__":
    anno = pd.read_csv(TRAIN_CSV_PATH)

    # Specify the image base path if different from the default
    image_base_path = DEFAULT_IMAGE_BASE_PATH

    output_file = OUTPUT_FILE_PATH

    open(output_file, FILE_WRITE_MODE_TRUNCATE).close()
    print(START_MESSAGE)
    parallel_process_images(anno, image_base_path, output_file, batch_size=MAIN_BATCH_SIZE)

    # new_anno.to_csv("imaterialist-fashion-2020-fgvc7/new_annotations.csv", index=False)
