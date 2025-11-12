import json

from objects.tile import Tile

def get_board_from_data(data):
    return Board(
        height=data['grid_size']['height'],
        width=data['grid_size']['width'],
        difficulty_level=data['difficulty_level'],
        difficulty_score=data['difficulty_score'],
        id=data['id'],
        polyshapes=data['polyshapes'],
        puzzle_array=data['puzzle_array'],
        solution_count=data['solution_count'],
        solutions=data['solutions']
    )


class Board:
    def __init__(self, height, width, difficulty_level, difficulty_score, id, polyshapes, puzzle_array, solution_count, solutions):
        self.height = height * 2 + 1
        self.width = width * 2 + 1
        self.difficulty_level = difficulty_level
        self.difficulty_score = difficulty_score
        self.id = id
        self.polyshapes = json.loads(polyshapes)
        self.puzzle_array = puzzle_array
        self.solution_count = solution_count
        self.solutions = solutions
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.place_all_objects()

    def place_object(self, x, y, obj):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = obj
        else:
            raise ValueError("Coordinates out of bounds")

    def place_all_objects(self):
        for y in range(self.height):
            for x in range(self.width):
                obj_abbr = self.puzzle_array[y][x]
                obj = Tile(obj_abbr)
                self.place_object(x, y, obj)

    def get_object(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        else:
            raise ValueError("Coordinates out of bounds")