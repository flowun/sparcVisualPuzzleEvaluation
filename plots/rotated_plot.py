from PIL import Image, ImageChops
import math

from plots.original_plot import OriginalPlot
from plots.path_cell_annotation_plot import PathCellAnnotatedPlot

# work in progress
class RotatedPlot(OriginalPlot):
    def __init__(self, *args, angle=15, side_pixel_removal_factor=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_angle = angle
        self.side_pixel_removal_factor = side_pixel_removal_factor  # side_pixel_removal_factor * pad_pixels / 2 will be removed from each side

    def render(self):
        super().render()

        base = self._img
        target_w, target_h = base.size
        target_w -= int(self.side_pixel_removal_factor * self.pad_pixels)
        target_h -= int(self.side_pixel_removal_factor * self.pad_pixels)
        fill_color = self.background_color

        bg = Image.new(base.mode, (target_w, target_h), fill_color)
        diff = ImageChops.difference(base, bg)
        content_bbox = diff.getbbox() or (0, 0, target_w, target_h)
        content_left, content_top, content_right, content_bottom = content_bbox
        pad_left = content_left - int(self.side_pixel_removal_factor * self.pad_pixels)
        pad_top = content_top - int(self.side_pixel_removal_factor * self.pad_pixels)
        pad_right = target_w - content_right
        pad_bottom = target_h - content_bottom

        content = base.crop(content_bbox)
        content_w, content_h = content.size

        theta = math.radians(self.rotation_angle % 360)
        cos_t = abs(math.cos(theta))
        sin_t = abs(math.sin(theta))
        bbox_w = max(1, int(math.ceil(content_w * cos_t + content_h * sin_t)))
        bbox_h = max(1, int(math.ceil(content_w * sin_t + content_h * cos_t)))

        canvas = Image.new(base.mode, (bbox_w, bbox_h), fill_color)
        paste_x = (bbox_w - content_w) // 2
        paste_y = (bbox_h - content_h) // 2
        canvas.paste(content, (paste_x, paste_y))

        rotated = canvas.rotate(self.rotation_angle, resample=Image.BICUBIC, expand=False, fillcolor=fill_color)

        final_w = rotated.width + pad_left + pad_right
        final_h = rotated.height + pad_top + pad_bottom
        result = Image.new(base.mode, (final_w, final_h), fill_color)
        result.paste(rotated, (pad_left, pad_top))
        self._img = result
        return


class RotatedPathCellAnnotatedPlot(RotatedPlot, PathCellAnnotatedPlot):
    pass


if __name__ == "__main__":
    import os
    from datasets import load_dataset, load_from_disk
    from objects import board

    if "sparc_text_dataset" not in os.listdir("../data"):
        dataset = load_dataset("lkaesberg/SPaRC", "all", split=None)
        dataset.save_to_disk("../data/sparc_text_dataset")

    dataset = load_from_disk("../data/sparc_text_dataset")["test"]
    idx = next(i for i, d in enumerate(dataset) if d['id'] == "b4d66071dc728f44")

    b1 = board.Board(
        height=dataset[idx]['grid_size']['height'],
        width=dataset[idx]['grid_size']['width'],
        difficulty_level=dataset[idx]['difficulty_level'],
        difficulty_score=dataset[idx]['difficulty_score'],
        id=dataset[idx]['id'],
        polyshapes=dataset[idx]['polyshapes'],
        puzzle_array=dataset[idx]['puzzle_array'],
        solution_count=dataset[idx]['solution_count'],
        solutions=dataset[idx]['solutions']
    )

    plot = RotatedPathCellAnnotatedPlot(
        board=b1,
        size=(b1.width, b1.height),
        angle=15,
        side_pixel_removal_factor=0.2
    )
    plot.render()
    plot._img.show()