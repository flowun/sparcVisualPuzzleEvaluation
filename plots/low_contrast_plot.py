from plots.original_plot import OriginalPlot
from plots.path_cell_annotation_plot import PathCellAnnotatedPlot


class LowContrastPlot(OriginalPlot):
    def __init__(self, board, size=(-1, -1), **kwargs):
        super().__init__(board, size=size)
        # Override background and cell colors for low contrast (same as done for the low_contrast mapping in the _color_from_code method)
        background_gray = int(0.3 * self.background_color[0] + 0.59 * self.background_color[1] + 0.11 * self.background_color[2])
        self.background_color = (
            int((self.background_color[0] + background_gray) / 2),
            int((self.background_color[1] + background_gray) / 2),
            int((self.background_color[2] + background_gray) / 2),
        )
        cell_gray = int(0.3 * self.cell_color[0] + 0.59 * self.cell_color[1] + 0.11 * self.cell_color[2])
        self.cell_color = (
            int((self.cell_color[0] + cell_gray) / 2),
            int((self.cell_color[1] + cell_gray) / 2),
            int((self.cell_color[2] + cell_gray) / 2),
        )
        path_color_gray = int(0.3 * self.path_color[0] + 0.59 * self.path_color[1] + 0.11 * self.path_color[2])
        self.path_color = (
            int((self.path_color[0] + path_color_gray) / 2),
            int((self.path_color[1] + path_color_gray) / 2),
            int((self.path_color[2] + path_color_gray) / 2),
        )

    def _color_from_code(self, code):
        default_mapping = {
            'R': (234, 51, 34),   # Red
            'B': (0, 0, 245),  # Blue
            'G': (55, 126, 34),  # Green
            'Y': (255, 255, 84),  # Yellow
            'W': (236, 240, 241), # White
            'O': (242, 169, 59),  # Orange
            'P': (117, 20, 124),  # Purple
            'K': (0, 0, 0),       # Black
            None: (0, 0, 0),
        }
        # Adjust colors more into gray scale direction for low contrast
        low_contrast_mapping = default_mapping.copy()
        for key, (r, g, b) in default_mapping.items():
            gray = int(0.3 * r + 0.59 * g + 0.11 * b)
            adjusted_r = int((r + gray) / 2)
            adjusted_g = int((g + gray) / 2)
            adjusted_b = int((b + gray) / 2)
            low_contrast_mapping[key] = (adjusted_r, adjusted_g, adjusted_b)
        return low_contrast_mapping.get(code, (0, 0, 0))


class LowContrastPathCellAnnotatedPlot(PathCellAnnotatedPlot, LowContrastPlot):
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

    idx = 3
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

    plot = LowContrastPathCellAnnotatedPlot(
        board=b1,
        size=(b1.width, b1.height),
    )
    plot.render()
    plot._img.show()