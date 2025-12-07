from PIL import Image

from plots.original_plot import OriginalPlot
from plots.path_cell_annotation_plot import PathCellAnnotatedPlot


class BlackFramePlot(OriginalPlot):
    # Same as OriginalPlot but extends the image after it is rendered with an additional black border
    def __init__(self,
                 board,
                 extra_border_pixels=20,
                 **kwargs,
                 ):
        super().__init__(board, **kwargs)
        self.extra_border_pixels = extra_border_pixels

    def render(self):
        super().render()
        new_x_size = self.x_image_size + 2 * self.extra_border_pixels
        new_y_size = self.y_image_size + 2 * self.extra_border_pixels
        new_img = Image.new("RGB", (new_x_size, new_y_size), (0, 0, 0))  # black background
        new_img.paste(self._img, (self.extra_border_pixels, self.extra_border_pixels))
        self._img = new_img


class BlackFramePathCellAnnotatedPlot(BlackFramePlot, PathCellAnnotatedPlot):
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

    # idx = 0
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

    plot = BlackFramePathCellAnnotatedPlot(
        board=b1,
        size=(b1.width, b1.height),
    )
    plot.render()
    plot._img.show()