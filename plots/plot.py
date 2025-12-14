import os
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw

def get_plot_class(plot_type, board, **kwargs):
    if plot_type == "original":
        from plots.original_plot import OriginalPlot
        return OriginalPlot(board, **kwargs)
    elif plot_type == "start_end_marked":
        from plots.start_end_marked_plot import StartEndMarkedPlot
        return StartEndMarkedPlot(board, **kwargs)
    elif plot_type == "coordinate_grid":
        from plots.coordinate_grid_plot import CoordinateGridPlot
        return CoordinateGridPlot(board, **kwargs)
    elif plot_type == "coordinate_grid_and_start_end_marked":
        from plots.coordinate_grid_plot import CoordinateGridAndStartEndMarkedPlot
        return CoordinateGridAndStartEndMarkedPlot(board, **kwargs)
    elif plot_type == "path_cell_annotated":
        from plots.path_cell_annotation_plot import PathCellAnnotatedPlot
        return PathCellAnnotatedPlot(board, **kwargs)
    elif plot_type == "text":
        from plots.text_plot import TextPlot
        return TextPlot(board, **kwargs)
    elif plot_type == "low_contrast":
        from plots.low_contrast_plot import LowContrastPlot
        return LowContrastPlot(board, **kwargs)
    elif plot_type == "low_contrast_and_path_cell_annotated":
        from plots.low_contrast_plot import LowContrastPathCellAnnotatedPlot
        return LowContrastPathCellAnnotatedPlot(board, **kwargs)
    elif plot_type == "low_resolution":
        from plots.low_resolution_plot import LowResolutionPlot
        return LowResolutionPlot(board, **kwargs)
    elif plot_type == "low_resolution_and_path_cell_annotated":
        from plots.low_resolution_plot import LowResolutionPathCellAnnotatedPlot
        return LowResolutionPathCellAnnotatedPlot(board, **kwargs)
    elif plot_type == "rotated":
        from plots.rotated_plot import RotatedPlot
        return RotatedPlot(board, **kwargs)
    elif plot_type == "rotated_and_path_cell_annotated":
        from plots.rotated_plot import RotatedPathCellAnnotatedPlot
        return RotatedPathCellAnnotatedPlot(board, **kwargs)
    elif plot_type == "black_frame":
        from plots.black_frame_plot import BlackFramePlot
        return BlackFramePlot(board, **kwargs)
    elif plot_type == "black_frame_and_path_cell_annotated":
        from plots.black_frame_plot import BlackFramePathCellAnnotatedPlot
        return BlackFramePathCellAnnotatedPlot(board, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


class Plot(ABC):
    def __init__(
            self,
            board,
            path_cell_size_ratio=(2, 5),
            size=(-1, -1),  # board size in number of fields
            cell_size=100,
            background_color=(255, 255, 255),
            cell_color=(200, 200, 200),
            path_color=(150, 150, 150),
            pad_pixels=30,
    ):
        super().__init__()
        if size[0] == -1 or size[1] == -1:
            raise ValueError("Size must be specified and cannot be -1")
        self.board = board
        self.path_cell_size_ratio = path_cell_size_ratio
        self.width = size[0]
        self.height = size[1]
        self.cell_size = cell_size  # Size of each cell in pixels/units
        self.path_size = cell_size * path_cell_size_ratio[0] / path_cell_size_ratio[1]
        self.background_color = background_color
        self.cell_color = cell_color
        self.path_color = path_color
        self.pad_pixels = pad_pixels
        self.positions = [[(0, 0, 0, 0) for _ in range(board.width)] for _ in range(board.height)]  # top-left x, top-left y, bottom-right x, bottom-right y
        self.middle_positions = [[(0, 0) for _ in range(board.width)] for _ in range(board.height)]  # middle x, middle y
        self.x_image_size, self.y_image_size = 0, 0
        self.calculate_positions()
        self._img = None
        self._draw = None

    def calculate_positions(self):
        # calculates top right pixel positions for each cell in the board
        current_pos_y, current_pos_x = self.pad_pixels + 1, self.pad_pixels + 1
        for y in range(self.board.height):
            if y % 2 == 0:
                next_pos_y = current_pos_y + self.path_size
            else:
                next_pos_y = current_pos_y + self.cell_size

            current_pos_x = self.pad_pixels + 1
            for x in range(self.board.width):
                if x % 2 == 0:
                    next_pos_x = current_pos_x + self.path_size
                else:
                    next_pos_x = current_pos_x + self.cell_size
                self.positions[y][x] = (current_pos_x, current_pos_y, next_pos_x - 1, next_pos_y - 1)
                self.middle_positions[y][x] = (int((current_pos_x + next_pos_x - 1) / 2), int((current_pos_y + next_pos_y - 1) / 2))
                current_pos_x = next_pos_x
            current_pos_y = next_pos_y
        self.x_image_size = int((current_pos_x - 1) + self.pad_pixels)
        self.y_image_size = int((current_pos_y - 1) + self.pad_pixels)

    @abstractmethod
    def render(self):
        self._img = Image.new("RGB", (self.x_image_size, self.y_image_size), self.background_color)
        self._draw = ImageDraw.Draw(self._img)

    def overlay_path(self, path, path_color=(255, 255, 255)):
        if self._img is None or self._draw is None:
            raise RuntimeError("Nothing to overlay path on. Call render(...) first.")
        if not path or len(path) < 2:
            return
        if path_color is None:
            path_color = self.path_color

        def _to_xy(cell):
            if isinstance(cell, dict):
                return int(cell.get("x")), int(cell.get("y"))
            try:
                return int(cell[0]), int(cell[1])
            except Exception:
                raise ValueError(f"Unsupported path cell format: {cell!r}")

        def signum(x):
            return (x > 0) - (x < 0)

        line_width = int(self.path_size) // 4

        # circle at start
        start_x, start_y = _to_xy(path[0])
        start_cx, start_cy = self.middle_positions[start_y][start_x]
        radius = max(1, line_width * 3 // 2)
        self._draw.ellipse(
            (start_cx - radius, start_cy - radius, start_cx + radius, start_cy + radius),
            fill=path_color,
            outline=path_color,
            width=max(1, line_width // 2),
        )

        for i in range(len(path) - 1):
            sx, sy = _to_xy(path[i])
            ex, ey = _to_xy(path[i + 1])
            start_pos = list(self.middle_positions[sy][sx])
            end_pos = list(self.middle_positions[ey][ex])
            start_pos[0] += int(signum(sx - ex) * (line_width / 2 - 1))
            end_pos[0] += int(signum(ex - sx) * (line_width / 2 - 1))
            start_pos[1] += int(signum(sy - ey) * (line_width / 2 - 1))
            end_pos[1] += int(signum(ey - sy) * (line_width / 2 - 1))
            self._draw.line([start_pos, end_pos], fill=path_color, width=line_width)

    def save(self, dir, filename):
        if self._img is None:
            raise RuntimeError("Nothing to save. Call render(...) first.")
        # Fast PNG writing: low compression and no optimize pass.
        if not os.path.exists(dir):
            os.makedirs(f"{dir}")
        self._img.save(f"{dir}/{filename}", format="PNG", compress_level=1, optimize=False)

    def show(self):
        if self._img is None:
            raise RuntimeError("Nothing to show. Call render(...) first.")
        self._img.show()
