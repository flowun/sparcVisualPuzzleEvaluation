from PIL import Image, ImageDraw, ImageFont
import math

from plots.plot import Plot

class OriginalPlot(Plot):
    def __init__(self,
                 board,
                 path_cell_size_ratio=(2, 5),
                 size=(-1, -1),  # board size in number of fields
                 cell_size=100,
                 background_color=(28, 55, 51),
                 cell_color=(75, 167, 138),
                 path_color=(54, 68, 68),
                 pad_pixels=50,
                 ):
        super().__init__(
            board,
            path_cell_size_ratio,
            size,
            cell_size,
            background_color,
            cell_color,
            path_color,
            pad_pixels,
        )
        # Configurable size ratios
        self.dot_ratio = 0.8           # of self.path_size
        self.stone_ratio = 0.5         # of self.cell_size
        self.star_ratio = 0.55         # of self.cell_size
        self.triangle_ratio = 0.28     # side length as ratio of self.cell_size
        self.triangle_gap_ratio = 0.2  # gap between triangles as ratio of triangle side
        self.polyshape_ratio = 0.9              # total grid size as ratio of self.cell_size
        self.polyshape_outline_ratio = 0.2      # outline width as ratio of unit square size
        self.polyshape_gap_ratio = 0.3          # gap between unit squares as ratio of unit size
        # Start/End/corners
        self.start_radius_ratio = 1.0          # Start circle radius as ratio of path_size
        self.end_extension_ratio = 1.0          # End extension length as ratio of path_size
        self.end_cap_round_ratio = 0.5         # End extension rounding radius as ratio of path_size

    def _color_from_code(self, code):
        mapping = {
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
        return mapping.get(code, (0, 0, 0))

    def initialize_image(self):
        self._img = Image.new("RGB", (self.x_image_size, self.y_image_size), self.background_color)
        self._draw = ImageDraw.Draw(self._img)


    def render(self):
        self.initialize_image()

        # # # # # # # # # # # # # # # # # # #
        # Board structure (without pieces)  #
        # # # # # # # # # # # # # # # # # # #

        # cell background
        self._draw.rectangle([self.positions[0][0][0], self.positions[0][0][1], self.positions[-1][-1][2], self.positions[-1][-1][3]], fill=self.cell_color)

        for y in range(self.board.height):
            for x in range(self.board.width):
                obj = self.board.get_object(x, y)
                top_left_x = self.positions[y][x][0]
                top_left_y = self.positions[y][x][1]
                bottom_right_x = self.positions[y][x][2]
                bottom_right_y = self.positions[y][x][3]
                if x % 2 == 1 and y % 2 == 1:
                    # Cell
                    pass
                else:
                    # Path (that is not a corner)
                    orientation = "corner"
                    if x % 2 == 1 and y % 2 == 0:
                        orientation = "horizontal"
                    elif x % 2 == 0 and y % 2 == 1:
                        orientation = "vertical"
                    if obj.type == "Gap":
                        # 1/3 fill, 1/3 empty, 1/3 fill
                        if orientation == "horizontal":
                            third_width = (bottom_right_x - top_left_x + 1) / 3
                            self._draw.rectangle([top_left_x, top_left_y, top_left_x + third_width - 1, bottom_right_y], fill=self.path_color)
                            self._draw.rectangle([top_left_x + 2 * third_width, top_left_y, bottom_right_x, bottom_right_y], fill=self.path_color)
                        elif orientation == "vertical":
                            third_height = (bottom_right_y - top_left_y + 1) / 3
                            self._draw.rectangle([top_left_x, top_left_y, bottom_right_x, top_left_y + third_height - 1], fill=self.path_color)
                            self._draw.rectangle([top_left_x, top_left_y + 2 * third_height, bottom_right_x, bottom_right_y], fill=self.path_color)
                        else:
                            raise ValueError("Corner gaps not supported")
                    elif x in (0, self.board.width - 1) and y in (0, self.board.height - 1) and not obj.type == "End":  # Corners are filled with background instead to make them round later
                        self._draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=self.background_color)
                    else:
                        # Full fill
                        self._draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=self.path_color)
                    # Round the outer board corners
                    if x == 0 and y == 0:
                        tl = self.positions[0][0]
                        self._draw.pieslice([tl[0], tl[1], tl[0] + 2 * self.path_size, tl[1] + 2 * self.path_size], start=180, end=270, fill=self.path_color)
                    elif x == self.board.width - 1 and y == 0:
                        tr = self.positions[0][self.board.width - 1]
                        self._draw.pieslice([tr[0] - self.path_size, tr[1], tr[0] + self.path_size, tr[1] + 2 * self.path_size], start=270, end=360, fill=self.path_color)
                    elif x == 0 and y == self.board.height - 1:
                        bl = self.positions[self.board.height - 1][0]
                        self._draw.pieslice([bl[0], bl[1] - self.path_size, bl[0] + 2 * self.path_size, bl[1] + self.path_size], start=90, end=180, fill=self.path_color)
                    elif x == self.board.width - 1 and y == self.board.height - 1:
                        br = self.positions[self.board.height - 1][self.board.width - 1]
                        self._draw.pieslice([br[0] - self.path_size, br[1] - self.path_size, br[0] + self.path_size, br[1] + self.path_size], start=0, end=90, fill=self.path_color)

                # # # # # # # # #
                # Board pieces  #
                # # # # # # # # #
                cx, cy = self.middle_positions[y][x]
                if obj.type == "Start":
                    # Circle in the middle of the start field with path color
                    r = int(self.path_size * self.start_radius_ratio)
                    self._draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self.path_color)
                elif obj.type == "End":
                    # Extend outward by one path size with a rounded cap only on the outer side
                    ext_len = int(self.path_size * self.end_extension_ratio)
                    thick = int(round(self.path_size))
                    cap_r = int(self.path_size * self.end_cap_round_ratio)
                    cap_r = max(0, min(cap_r, ext_len))  # clamp radius

                    dx = 0
                    dy = 0
                    if y == 0:
                        dy = -1
                    elif y == self.board.height - 1:
                        dy = 1
                    elif x == 0:
                        dx = -1
                    elif x == self.board.width - 1:
                        dx = 1

                    if dy != 0:
                        x0 = int(round(cx - thick / 2))
                        x1 = int(round(cx + thick / 2))
                        if dy < 0:  # up
                            rect_top = top_left_y - ext_len + cap_r
                            rect_bottom = top_left_y - 1
                            if rect_top <= rect_bottom:
                                self._draw.rectangle([x0, rect_top, x1, rect_bottom], fill=self.path_color)
                            if cap_r > 0:
                                cyc = top_left_y - ext_len + cap_r
                                self._draw.pieslice([cx - cap_r, cyc - cap_r, cx + cap_r, cyc + cap_r], start=180, end=360, fill=self.path_color)
                        else:       # down
                            rect_top = bottom_right_y + 1
                            rect_bottom = bottom_right_y + ext_len - cap_r
                            if rect_top <= rect_bottom:
                                self._draw.rectangle([x0, rect_top, x1, rect_bottom], fill=self.path_color)
                            if cap_r > 0:
                                cyc = bottom_right_y + ext_len - cap_r
                                self._draw.pieslice([cx - cap_r, cyc - cap_r, cx + cap_r, cyc + cap_r], start=0, end=180, fill=self.path_color)
                    elif dx != 0:
                        y0 = int(round(cy - thick / 2))
                        y1 = int(round(cy + thick / 2))
                        if dx < 0:  # left
                            rect_left = top_left_x - ext_len + cap_r
                            rect_right = top_left_x - 1
                            if rect_left <= rect_right:
                                self._draw.rectangle([rect_left, y0, rect_right, y1], fill=self.path_color)
                            if cap_r > 0:
                                cxc = top_left_x - ext_len + cap_r
                                self._draw.pieslice([cxc - cap_r, cy - cap_r, cxc + cap_r, cy + cap_r], start=90, end=270, fill=self.path_color)
                        else:       # right
                            rect_left = bottom_right_x + 1
                            rect_right = bottom_right_x + ext_len - cap_r
                            if rect_left <= rect_right:
                                self._draw.rectangle([rect_left, y0, rect_right, y1], fill=self.path_color)
                            if cap_r > 0:
                                cxc = bottom_right_x + ext_len - cap_r
                                self._draw.pieslice([cxc - cap_r, cy - cap_r, cxc + cap_r, cy + cap_r], start=270, end=90, fill=self.path_color)
                elif obj.type == "Dot":
                    # Black hexagon on paths, sized by self.path_size
                    half_w = (bottom_right_x - top_left_x + 1) / 2
                    half_h = (bottom_right_y - top_left_y + 1) / 2
                    r = int(min(self.path_size * self.dot_ratio / 2, half_w - 1, half_h - 1))
                    if r > 1:
                        hex_pts = []
                        for i in range(6):
                            ang = math.radians(60 * i)
                            px = int(cx + r * math.cos(ang))
                            py = int(cy + r * math.sin(ang))
                            hex_pts.append((px, py))
                        self._draw.polygon(hex_pts, fill=(0, 0, 0))
                elif obj.type == "Stone":
                    # Rounded square in cell, colored by tile color
                    size = self.cell_size * self.stone_ratio
                    hw = int(size / 2)
                    bbox = [int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw)]
                    radius = int(size * 0.15)
                    if hasattr(self._draw, "rounded_rectangle"):
                        self._draw.rounded_rectangle(bbox, radius=radius, fill=self._color_from_code(obj.color))
                    else:
                        self._draw.rectangle(bbox, fill=self._color_from_code(obj.color))
                elif obj.type == "Star":
                    # 8-spike star (16 points alternating), colored by tile color
                    R = self.cell_size * self.star_ratio / 2
                    r = R * 0.65
                    star_pts = []
                    for i in range(16):
                        ang = math.radians((360 / 16) * i - 90)  # start at top
                        rad = R if i % 2 == 0 else r
                        px = int(cx + rad * math.cos(ang))
                        py = int(cy + rad * math.sin(ang))
                        star_pts.append((px, py))
                    self._draw.polygon(star_pts, fill=self._color_from_code(obj.color))
                elif obj.type == "Triangle":
                    # Arrange triangles nicely:
                    # - 1..2: horizontal (existing behavior)
                    # - 3: pyramid (1 top, 2 bottom)
                    # - 4: 2x2 grid
                    n = getattr(obj, "triangle_count", 1)
                    n = max(1, min(4, int(n)))
                    side = self.cell_size * self.triangle_ratio
                    gap = side * self.triangle_gap_ratio
                    h = side * math.sqrt(3) / 2.0
                    color = self._color_from_code(obj.color)

                    if n <= 2:
                        # Keep existing horizontal arrangement
                        total_w = n * side + (n - 1) * gap
                        start_center_x = cx - total_w / 2 + side / 2
                        for i in range(n):
                            xi = start_center_x + i * (side + gap)
                            apex = (int(round(xi)), int(round(cy - 2 * h / 3)))
                            left = (int(round(xi - side / 2)), int(round(cy + h / 3)))
                            right = (int(round(xi + side / 2)), int(round(cy + h / 3)))
                            self._draw.polygon([apex, left, right], fill=color)
                    else:
                        # n == 3 or 4: multi-row layout
                        rows = [1, 2] if n == 3 else [2, 2]  # top->bottom
                        R = len(rows)
                        row_widths = [r * side + (r - 1) * gap for r in rows]
                        group_w = max(row_widths)
                        group_h = R * h + (R - 1) * gap

                        # Center the whole group in the cell (by bounding box)
                        top = cy - group_h / 2.0
                        cy0 = top + 2 * h / 3.0  # center y of the top row

                        for ri, cols in enumerate(rows):
                            row_w = row_widths[ri]
                            y_center = cy0 + ri * (h + gap)
                            start_x = cx - row_w / 2.0 + side / 2.0
                            for ci in range(cols):
                                xi = start_x + ci * (side + gap)
                                apex = (int(round(xi)), int(round(y_center - 2 * h / 3.0)))
                                left = (int(round(xi - side / 2.0)), int(round(y_center + h / 3.0)))
                                right = (int(round(xi + side / 2.0)), int(round(y_center + h / 3.0)))
                                self._draw.polygon([apex, left, right], fill=color)
                elif obj.type == "Positive Polyshape" or obj.type == "Negative Polyshape":
                    shape_def = self.board.polyshapes[str(obj.shape_id)]
                    grid_size = self.cell_size * self.polyshape_ratio
                    unit = grid_size / 4.0
                    left = cx - grid_size / 2.0
                    top = cy - grid_size / 2.0
                    # Determine active rows/cols and center only the drawn subgrid
                    active_rows = [ry for ry in range(4) if any(shape_def[ry][rx] == 1 for rx in range(4))]
                    active_cols = [rx for rx in range(4) if any(shape_def[ry][rx] == 1 for ry in range(4))]
                    min_r, max_r = min(active_rows), max(active_rows)
                    min_c, max_c = min(active_cols), max(active_cols)
                    # Current center of active subgrid (without insets)
                    cur_cx = left + ((min_c + max_c + 1) * unit) / 2.0
                    cur_cy = top + ((min_r + max_r + 1) * unit) / 2.0
                    dx = cx - cur_cx
                    dy = cy - cur_cy
                    xs = [left + i * unit for i in range(5)]
                    ys = [top + i * unit for i in range(5)]
                    color = self._color_from_code(obj.color)
                    # Small gap between unit squares
                    gap = unit * self.polyshape_gap_ratio
                    inset = gap / 2.0
                    base_outline = max(1, int(unit * self.polyshape_outline_ratio))
                    for ry in range(4):
                        for rx in range(4):
                            if shape_def[ry][rx] == 1:
                                x0f = xs[rx] + inset + dx
                                y0f = ys[ry] + inset + dy
                                x1f = xs[rx + 1] - 1 - inset + dx
                                y1f = ys[ry + 1] - 1 - inset + dy
                                x0 = int(round(x0f))
                                y0 = int(round(y0f))
                                x1 = int(round(x1f))
                                y1 = int(round(y1f))
                                if obj.type == "Positive Polyshape":
                                    self._draw.rectangle([x0, y0, x1, y1], fill=color)
                                else:
                                    # Negative Polyshape: hollow square (see-through inside)
                                    inner_span = max(1, min(x1 - x0 + 1, y1 - y0 + 1))
                                    outline_w = max(1, min(base_outline, inner_span // 3))
                                    self._draw.rectangle([x0, y0, x1, y1], outline=color, width=outline_w)

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


class CoordinateGridPlot(OriginalPlot):
    def __init__(self,
                 board,
                 size=(-1, -1),
                 coordinate_color=(255, 255, 255),
                 coordinate_font_size=20,
                 pad_pixels=50,
                 extra_coordinate_info_pad_pixels=50,
                 grid_lines=True
                 ):
        super().__init__(board, size=size, pad_pixels=pad_pixels+extra_coordinate_info_pad_pixels)
        self.coordinate_color = coordinate_color
        self.coordinate_font_size = coordinate_font_size
        self.extra_coordinate_info_pad_pixels = extra_coordinate_info_pad_pixels
        self.grid_lines = grid_lines

    def initialize_image(self):
        self.x_image_size -= self.extra_coordinate_info_pad_pixels
        self.y_image_size -= self.extra_coordinate_info_pad_pixels
        self._img = Image.new("RGB", (self.x_image_size, self.y_image_size), self.background_color)
        self._draw = ImageDraw.Draw(self._img)

    def render(self):
        super().render()
        # Draw coordinates on top and left side
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

            self._draw.text((self.positions[0][0][0] * 1/3, self.positions[0][0][1] * 2/3), "y", fill=self.coordinate_color, font_size=self.coordinate_font_size)
            self._draw.text((self.positions[0][0][0] * 2/3, self.positions[0][0][1] * 1/3), "x", fill=self.coordinate_color, font_size=self.coordinate_font_size)


class CoordinateGridAndStartEndMarkedPlot(CoordinateGridPlot, StartEndMarkedPlot):
    def render(self):
        super().render()


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