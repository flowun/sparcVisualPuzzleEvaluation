from PIL import Image, ImageDraw

from plots.original_plot import OriginalPlot
from plots.start_end_marked_plot import StartEndMarkedPlot


class CoordinateGridPlot(OriginalPlot):
    def __init__(self,
                 board,
                 size=(-1, -1),
                 coordinate_color=(255, 255, 255),
                 coordinate_font_size=20,
                 pad_pixels=50,
                 extra_coordinate_info_pad_pixels=50,
                 grid_lines=True,
                 show_labels=True
                 ):
        super().__init__(board, size=size, pad_pixels=pad_pixels+extra_coordinate_info_pad_pixels)
        self.coordinate_color = coordinate_color
        self.coordinate_font_size = coordinate_font_size
        self.extra_coordinate_info_pad_pixels = extra_coordinate_info_pad_pixels
        self.grid_lines = grid_lines
        self.show_labels = show_labels

    def initialize_image(self):
        self.x_image_size -= self.extra_coordinate_info_pad_pixels
        self.y_image_size -= self.extra_coordinate_info_pad_pixels
        self._img = Image.new("RGB", (self.x_image_size, self.y_image_size), self.background_color)
        self._draw = ImageDraw.Draw(self._img)

    def render(self):
        super().render()
        # Draw coordinates on top and left side
        if self.show_labels:
            for x in range(self.board.width):
                cx, cy = self.middle_positions[0][x]
                self._draw.text((cx - self.coordinate_font_size // 4, self.positions[0][0][1] * 1/3), str(x), fill=self.coordinate_color, font_size=self.coordinate_font_size)
            for y in range(self.board.height):
                cx, cy = self.middle_positions[y][0]
                self._draw.text((self.positions[0][0][0] * 1/3, cy - self.coordinate_font_size // 2), str(y), fill=self.coordinate_color, font_size=self.coordinate_font_size)
        # Draw grid lines if enabled
        if self.grid_lines:
            # Vertical lines (extend to full image height)
            for x in range(self.board.width):
                x_line = self.positions[0][x][0]
                self._draw.line([x_line, 0, x_line, self.y_image_size - 1], fill=self.coordinate_color, width=1)
            # Rightmost border line
            right_edge_x = self.positions[0][-1][2]
            self._draw.line([right_edge_x, 0, right_edge_x, self.y_image_size - 1], fill=self.coordinate_color, width=1)

            # Horizontal lines (extend to full image width)
            for y in range(self.board.height):
                y_line = self.positions[y][0][1]
                self._draw.line([0, y_line, self.x_image_size - 1, y_line], fill=self.coordinate_color, width=1)
            # Bottom border line
            bottom_edge_y = self.positions[-1][0][3]
            self._draw.line([0, bottom_edge_y, self.x_image_size - 1, bottom_edge_y], fill=self.coordinate_color, width=1)

            self._draw.line([0, 0, self.positions[0][0][0], self.positions[0][0][1]], fill=self.coordinate_color, width=1)

            if self.show_labels:
                self._draw.text((self.positions[0][0][0] * 1/3, self.positions[0][0][1] * 2/3), "y", fill=self.coordinate_color, font_size=self.coordinate_font_size)
                self._draw.text((self.positions[0][0][0] * 2/3, self.positions[0][0][1] * 1/3), "x", fill=self.coordinate_color, font_size=self.coordinate_font_size)


class CoordinateGridAndStartEndMarkedPlot(CoordinateGridPlot, StartEndMarkedPlot):
    def render(self):
        super().render()


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

    plot = CoordinateGridAndStartEndMarkedPlot(
        board=b1,
        size=(b1.width, b1.height),
    )
    plot.render()
    plot._img.show()