from PIL import Image

from plots.original_plot import OriginalPlot
from plots.path_cell_annotation_plot import PathCellAnnotatedPlot

class LowResolutionPlot(OriginalPlot):
    def __init__(self, *args, scale=2, upscale_for_display=True, **kwargs):
        """
        scale: integer downscale factor (2 = half resolution, 4 = quarter, etc.)
        upscale_for_display: if True, the image is upscaled back to original size using
                             nearest neighbor so it appears pixelated but keeps display size.
        """
        super().__init__(*args, **kwargs)
        self._downscale = max(1, scale)
        self._upscale_display = upscale_for_display

    def render(self):
        # call parent render which produces self._img (PIL Image)
        super().render()

        w, h = self._img.size
        new_size = (max(1, w // self._downscale), max(1, h // self._downscale))

        # downsample to lower resolution
        low_res = self._img.resize(new_size, resample=Image.BILINEAR)

        if self._upscale_display:
            # keep original window size but show blocky pixels
            self._img = low_res.resize((w, h), resample=Image.NEAREST)
        else:
            # keep physically lower resolution image
            self._img = low_res


class LowResolutionPathCellAnnotatedPlot(LowResolutionPlot, PathCellAnnotatedPlot):
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

    idx = 5
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

    plot = LowResolutionPathCellAnnotatedPlot(
        board=b1,
        size=(b1.width, b1.height),
        scale=2
    )
    plot.render()
    plot._img.show()