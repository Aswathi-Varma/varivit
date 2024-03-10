import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Optional, Union

class Pix2StructImageProcessor(nn.Module):
    r"""
    Constructs a Pix2Struct image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
            deviation.
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
        max_patches (`int`, *optional*, defaults to 2048):
            The maximum number of patches to extract from the image as per the [Pix2Struct
            paper](https://arxiv.org/pdf/2210.03347.pdf).
        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether or not the image processor is for the VQA task. If `True` and `header_text` is passed in, text is
            rendered onto the input images.
    """

    model_input_names = ["flattened_patches"]

    def __init__(
        self,
        patch_size: Dict[str, int] = None,
        max_patches: int = 2048,
    ):
        super().__init__()
        self.patch_size = patch_size if patch_size is not None else {"length": 16,"height": 16, "width": 16}
        self.max_patches = max_patches

    def extract_flattened_patches(
        self,
        image: np.ndarray,
        max_patches: int,
        patch_size: dict,
    ):
        """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.

        Returns:
            result (`np.ndarray`):
                A sequence of `max_patches` flattened patches.
        """
        patch_length, patch_height, patch_width = patch_size["length"], patch_size["height"], patch_size["width"]
        image_length, image_height, image_width = 0,0,0

        # # maximize scale s.t.
        # scale = math.sqrt(max_patches * (patch_length / image_height)*(patch_height / image_height) * (patch_width / image_width))
        # num_feasible_l = max(min(math.floor(scale * image_length / patch_length), max_patches), 1)
        # num_feasible_h = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        # num_feasible_w = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)

        # resized_length = max(num_feasible_l * patch_length, 1)
        # resized_height = max(num_feasible_h * patch_height, 1)
        # resized_width = max(num_feasible_w * patch_width, 1)

        # image = torch.nn.functional.interpolate(
        #     image.unsqueeze(0),
        #     size=(resized_length, resized_height, resized_width),
        #     mode="trilinear",
        #     align_corners=False,
        #     antialias=True,
        # ).squeeze(0)

        # # [1, rows, columns, patch_height * patch_width * image_channels]
        # patches = torch_extract_patches(image, patch_length, patch_height, patch_width)

        # patches_shape = patches.shape
        # len = patches_shape[1]
        # hgt = patches_shape[2]
        # wdt = patches_shape[3]

        # depth = patches_shape[4]

        # # [rows * columns, patch_height * patch_width * image_channels]
        # patches = patches.reshape([len * hgt * wdt, depth])

        # # [rows * columns, 1]
        # len_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        # hgt_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # # Offset by 1 so the ids do not contain zeros, which represent padding.
        # row_ids += 1
        # col_ids += 1

        # # Prepare additional patch features.
        # # [rows * columns, 1]
        # row_ids = row_ids.to(torch.float32)
        # col_ids = col_ids.to(torch.float32)

        # # [rows * columns, 2 + patch_height * patch_width * image_channels]
        # result = torch.cat([row_ids, col_ids, patches], -1)

        # # [max_patches, 2 + patch_height * patch_width * image_channels]
        # result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        # result = to_numpy_array(result)

        # return result
    
    # adapted from: https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2
def torch_extract_patches(image_tensor, patch_length, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    #(batch_size, channels, length, height, width)
    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_length, patch_height, patch_width), stride=(patch_length, patch_height, patch_width))
    
    patches = patches.reshape(
    image_tensor.size(0),
    image_tensor.size(1),
    patch_length,
    patch_height,
    patch_width,
    -1
    )   

    # Permute the dimensions to (batch_size, -1, patch_height, patch_width, patch_depth, channels)
    patches = patches.permute(0, 5, 3, 4, 2, 1)

    patches = patches.reshape(
        image_tensor.size(2) // patch_length,
        image_tensor.size(3) // patch_height,
        image_tensor.size(4) // patch_width,
        image_tensor.size(1) * patch_height * patch_width * patch_length,
    )
    return patches.unsqueeze(0)
