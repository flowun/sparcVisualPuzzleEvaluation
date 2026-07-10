from plots.original_plot import OriginalPlot


class GridLinesOnlyPlot(OriginalPlot):
    def __init__(self,
                 board,
                 size=(-1, -1),
                 grid_line_color=(255, 255, 255)
                 ):
        super().__init__(board, size=size)
        self.grid_line_color = grid_line_color

    def render(self):
        super().render()
        # Grid lines as in CoordinateGridPlot, but clipped to the board area:
        # no label gutter, no corner diagonal, no text
        left = self.positions[0][0][0]
        top = self.positions[0][0][1]
        right = self.positions[0][-1][2]
        bottom = self.positions[-1][0][3]
        # Vertical lines at every column boundary
        for x in range(self.board.width):
            x_line = self.positions[0][x][0]
            self._draw.line([x_line, top, x_line, bottom], fill=self.grid_line_color, width=1)
        self._draw.line([right, top, right, bottom], fill=self.grid_line_color, width=1)
        # Horizontal lines at every row boundary
        for y in range(self.board.height):
            y_line = self.positions[y][0][1]
            self._draw.line([left, y_line, right, y_line], fill=self.grid_line_color, width=1)
        self._draw.line([left, bottom, right, bottom], fill=self.grid_line_color, width=1)
