from PIL import Image, ImageDraw, ImageFont

from plots.plot import Plot


class TextPlot(Plot):
    def __init__(
        self,
        board,
        path_cell_size_ratio=(5, 10),
        size=(-1, -1),
        cell_size=100,
        background_color=(28, 55, 51),
        cell_color=(75, 167, 138),
        path_color=(54, 68, 68),
        pad_pixels=0,
        rule_text_color=(0, 0, 0),
        coordinate_color=(255, 255, 255),
        text_font_size=14,  # controls all text sizes
        font_path=None,
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
        self.rule_text_color = rule_text_color
        self.coordinate_color = coordinate_color
        self.text_font_size = text_font_size  # if None, computed from cell_size
        self.font_path = font_path
        self.font = ImageFont.truetype("arial.ttf", size=self.text_font_size)

    # Helper: color code to human-readable name
    def _color_name(self, code):
        mapping = {
            'R': 'Red',
            'B': 'Blue',
            'G': 'Green',
            'Y': 'Yellow',
            'W': 'White',
            'O': 'Orange',
            'P': 'Purple',
            'K': 'Black',
            None: '',
        }
        return mapping.get(code, '')

    def _rule_label(self, obj):
        t = getattr(obj, 'type', '')
        color_name = self._color_name(getattr(obj, 'color', None))
        if t == 'Stone':
            return f"{(color_name + '\n') if color_name else ''}Square".strip()
        if t == 'Star':
            return f"{(color_name + '\n') if color_name else ''}Star".strip()
        if t == 'Triangle':
            n = getattr(obj, 'triangle_count', 1)
            noun = 'Triangle' if int(n) == 1 else 'Triangles'
            return (f"{n} " + (f"{color_name}" if color_name else '') + f"\n{noun}").strip()
        if t == 'Positive Polyshape':
            return (f"{color_name + ' ' if color_name else ''}Positive\nPolyshape").strip()
        if t == 'Negative Polyshape':
            return (f"{color_name + ' ' if color_name else ''}Negative\nPolyshape").strip()
        if t == 'Empty Rule':
            return ""
        return (f"{(color_name + '\n') if color_name else ''}{t}").strip()

    def _draw_polyshape_under_text(self, obj, cx, cy):
        # 4x4 polyshape grid beneath the text, in black; filled for Positive, outlined for Negative
        shape_def = self.board.polyshapes[str(obj.shape_id)]
        shape_ratio = 0.60  # portion of cell height used for shape grid
        gap_ratio = 0.30    # between unit squares (as in original)
        outline_ratio = 0.20
        # Position shape slightly below center (to be under the text)
        grid_size = self.cell_size * shape_ratio
        unit = grid_size / 4.0
        # Vertical offset so there is room for the label above
        vertical_offset = self.cell_size * 0.15
        left = cx - grid_size / 2.0
        top = cy - grid_size / 2.0 + vertical_offset
        # active rows/cols to center drawn subgrid inside our grid box
        active_rows = [ry for ry in range(4) if any(shape_def[ry][rx] == 1 for rx in range(4))]
        active_cols = [rx for rx in range(4) if any(shape_def[ry][rx] == 1 for ry in range(4))]
        if not active_rows or not active_cols:
            return
        min_r, max_r = min(active_rows), max(active_rows)
        min_c, max_c = min(active_cols), max(active_cols)
        xs = [left + i * unit for i in range(5)]
        ys = [top + i * unit for i in range(5)]
        # Compute centering delta for active subgrid
        cur_cx = left + ((min_c + max_c + 1) * unit) / 2.0
        cur_cy = top + ((min_r + max_r + 1) * unit) / 2.0
        dx = cx - cur_cx
        dy = cy + vertical_offset - cur_cy  # center around shifted baseline
        gap = unit * gap_ratio
        inset = gap / 2.0
        base_outline = max(1, int(unit * outline_ratio))
        black = (0, 0, 0)
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
                    if obj.type == 'Positive Polyshape':
                        self._draw.rectangle([x0, y0, x1, y1], fill=black)
                    else:
                        inner_span = max(1, min(x1 - x0 + 1, y1 - y0 + 1))
                        outline_w = max(1, min(base_outline, inner_span // 3))
                        self._draw.rectangle([x0, y0, x1, y1], outline=black, width=outline_w)

    def render(self):
        self._img = Image.new("RGB", (self.x_image_size, self.y_image_size), self.background_color)
        self._draw = ImageDraw.Draw(self._img)

        # 1) Board background areas
        self._draw.rectangle(
            [
                self.positions[0][0][0],
                self.positions[0][0][1],
                self.positions[-1][-1][2],
                self.positions[-1][-1][3],
            ],
            fill=self.cell_color,
        )

        for y in range(self.board.height):
            for x in range(self.board.width):
                obj = self.board.get_object(x, y)
                tlx, tly, brx, bry = self.positions[y][x]
                if x % 2 == 1 and y % 2 == 1:
                    # Rule cell area already filled
                    pass
                else:
                    # Path cell or corner
                    orientation = "corner"
                    if x % 2 == 1 and y % 2 == 0:
                        orientation = "horizontal"
                    elif x % 2 == 0 and y % 2 == 1:
                        orientation = "vertical"

                    if getattr(obj, 'type', None) == "Gap":
                        # 1/3 fill, 1/3 empty, 1/3 fill
                        if orientation == "horizontal":
                            third_w = (brx - tlx + 1) / 3
                            self._draw.rectangle([tlx, tly, int(tlx + third_w - 1), bry], fill=self.path_color)
                            self._draw.rectangle([int(tlx + 2 * third_w), tly, brx, bry], fill=self.path_color)
                        elif orientation == "vertical":
                            third_h = (bry - tly + 1) / 3
                            self._draw.rectangle([tlx, tly, brx, int(tly + third_h - 1)], fill=self.path_color)
                            self._draw.rectangle([tlx, int(tly + 2 * third_h), brx, bry], fill=self.path_color)
                        else:
                            pass
                    else:
                        self._draw.rectangle([tlx, tly, brx, bry], fill=self.path_color)

                # 2) Textual content
                cx, cy = self.middle_positions[y][x]
                if x % 2 == 1 and y % 2 == 1:
                    # Rule cell label
                    label = self._rule_label(obj)
                    if label:
                        is_poly = getattr(obj, 'type', '') in ("Positive Polyshape", "Negative Polyshape")
                        if is_poly:
                            # Draw label slightly above center, then draw the polyshape under it
                            text_y = cy - int(self.cell_size * 0.22)
                            self._draw.multiline_text((cx, text_y), label, fill=self.rule_text_color, font=self.font, anchor="mm", align="center", spacing=2)
                            # Draw shape beneath the text
                            self._draw_polyshape_under_text(obj, cx, cy)
                        else:
                            self._draw.multiline_text((cx, cy), label, fill=self.rule_text_color, font=self.font, anchor="mm", align="center", spacing=2)
                else:
                    # Path cell: show coordinates except on gaps; Start/End have special labels
                    t = getattr(obj, 'type', None)
                    if t == 'Gap':
                        text = f"Gap"
                    elif t == 'Start':
                        text = f"Start\n({x},{y})"
                    elif t == 'End':
                        text = f"End\n({x},{y})"
                    elif t == 'Dot':
                        text = "Black\nDot"
                    else:
                        text = f"({x},{y})"
                    self._draw.multiline_text((cx, cy), text, fill=self.coordinate_color, font=self.font, anchor="mm", align="center", spacing=2)

        return self


if __name__ == "__main__":
    import os
    from datasets import load_dataset, load_from_disk
    from objects import board

    if "sparc_text_dataset" not in os.listdir("../data"):
        dataset = load_dataset("lkaesberg/SPaRC", "all", split=None)
        dataset.save_to_disk("../data/sparc_text_dataset")

    dataset = load_from_disk("../data/sparc_text_dataset")["test"]
    idx = next(i for i, d in enumerate(dataset) if d['id'] == "491bb36b613e4e47")

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

    plot = TextPlot(
        board=b1,
        size=(b1.width, b1.height),
    )
    plot.render()
    plot._img.show()