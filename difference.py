# script to find Difference Ratio
# Returns the difference ratio of two given images.

from __future__ import print_function
from PIL import Image, ImageChops, ImageStat

DIFF_IMG_FILE = "diff_img.png"


def diff(im1, im2, delete_diff_file=False, diff_img_file=None, ignore_alpha=False):
    """
    Calculate the difference between two images of the same size
    by comparing channel values at the pixel level.
    """

    if not diff_img_file:
        diff_img_file = DIFF_IMG_FILE

    # Ensure we have the same color channels (RGBA vs RGB)
    if im1.mode != im2.mode:
        raise ValueError(
            (
                "Differing color modes:\n  {}: {}\n  {}: {}\n"
                "Ensure image color modes are the same."
            ).format(im1_file, im1.mode, im2_file, im2.mode)
        )

    # Generate diff image in memory.
    diff_img = ImageChops.difference(im1, im2)

    if ignore_alpha:
        diff_img.putalpha(256)

    if not delete_diff_file:
        if "." not in diff_img_file:
            extension = "png"
        else:
            extension = diff_img_file.split(".")[-1]
        if extension in ("jpg", "jpeg"):
            extension = "jpeg"
            diff_img = diff_img.convert("RGB")
        diff_img.save(diff_img_file, extension)

    # Calculate difference as a ratio.
    stat = ImageStat.Stat(diff_img)
    removed_channels = 1 if ignore_alpha and len(stat.mean) == 4 else 0
    num_channels = len(stat.mean) - removed_channels
    sum_channel_values = sum(stat.mean[:num_channels])
    max_all_channels = num_channels * 255
    diff_ratio = sum_channel_values / max_all_channels

    return diff_ratio