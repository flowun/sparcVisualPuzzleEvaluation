from plots.original_plot import OriginalPlot

class StartEndMarkedPlot(OriginalPlot):
    def __init__(self,
                 board,
                 size=(-1, -1),
                 start_color=(0, 0, 0),
                 end_color=(0, 0, 0),
                 pad_pixels=50,
                 ):
        super().__init__(board, size=size, pad_pixels=pad_pixels)
        self.start_color = start_color
        self.end_color = end_color

    def render(self):
        super().render()
        # Highlight start and end positions
        for y in range(self.board.height):
            for x in range(self.board.width):
                obj = self.board.get_object(x, y)
                cx, cy = self.middle_positions[y][x]
                if obj.type == "Start":
                    self._draw.text((cx - self.cell_size // 6, cy - self.cell_size // 3.2), "S", fill=self.start_color, font_size=self.cell_size // 2)
                elif obj.type == "End":
                    self._draw.text((cx - self.cell_size // 6, cy - self.cell_size // 3.2), "E", fill=self.end_color, font_size=self.cell_size // 2)


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

    plot = StartEndMarkedPlot(
        board=b1,
        size=(b1.width, b1.height),
    )
    plot.render()
    plot._img.show()