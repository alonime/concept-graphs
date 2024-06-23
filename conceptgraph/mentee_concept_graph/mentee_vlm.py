import torch
from PIL import Image
import io
import base64

from conceptgraph.utils.vlm import get_image_captions_w_gpt4v




def image_captioning(image, openai_clent, detections, padding: int = 40):
    
    image = Image.fromarray(image)
    
    image_crops = []
    crops_captions = []
    
    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image_bytes = io.BytesIO()
        cropped_image.save(cropped_image_bytes, format='JPEG')
        encoded_cropped_image = base64.b64encode(cropped_image_bytes.getvalue()).decode('utf-8')

    
        captions = get_image_captions_w_gpt4v(openai_clent, encoded_cropped_image)
        crops_captions.append(captions)
        image_crops.append(cropped_image)

    
    return image_crops, crops_captions