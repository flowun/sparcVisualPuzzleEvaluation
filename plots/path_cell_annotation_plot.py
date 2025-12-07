from PIL import ImageFont

from plots.original_plot import OriginalPlot


class PathCellAnnotatedPlot(OriginalPlot):
    def __init__(self,
                 board,
                 size=(-1, -1),
                 pad_pixels=50,
                 path_cell_size_ratio=(5, 10),
                 ):
        super().__init__(board, size=size, pad_pixels=pad_pixels, path_cell_size_ratio=path_cell_size_ratio)
        self.font = ImageFont.truetype("arial.ttf", size=self.cell_size // 6)

    def render(self):
        super().render()
        for y in range(self.board.height):
            for x in range(self.board.width):
                obj = self.board.get_object(x, y)
                cx, cy = self.middle_positions[y][x]
                if not(x % 2 == 1 and y % 2 == 1):
                    if obj.type == 'Gap':
                        text = f"Gap"
                    elif obj.type == 'Start':
                        text = f"Start\n({x},{y})"
                    elif obj.type == 'End':
                        text = f"End\n({x},{y})"
                    elif obj.type == 'Dot':
                        text = ""
                    else:
                        text = f"({x},{y})"
                    self._draw.multiline_text((cx, cy), text, font=self.font, anchor="mm", align="center", spacing=2)


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

    plot = PathCellAnnotatedPlot(
        board=b1,
        size=(b1.width, b1.height),
    )
    plot.render()
    plot._img.show()