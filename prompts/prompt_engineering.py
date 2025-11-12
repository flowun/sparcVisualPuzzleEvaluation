
def get_prompt(data, board_type):
    return f"""## Objective
You are a specialized AI proficient in spatial reasoning and solving puzzles from the game ’The Witness’. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided image, adhering to all puzzle rules.

## Core Concepts & Grid Basics
* **Grid Dimensions:** The puzzle grid has {data['grid_size']['width'] * 2 + 1} columns and {data['grid_size']['height'] * 2 + 1} rows.
* **Coordinate System:** Nodes are identified by ‘(x, y)‘ coordinates. ‘(0,0)‘ is the top-left node. ‘({data['grid_size']['width'] * 2},{data['grid_size']['height'] * 2})‘ is the bottom-right node. ‘x‘ increases to the right, ‘y‘ increases downwards. Nodes can either be path cells or rule cells.
* **Path Cells:** Path cells are dark grey and are at all positions where at least one of ‘x‘ or ‘y‘ is even.
* **Path:** The solution is a single, continuous line connecting adjacent path cells either horizontally or vertically. Each step has a distance 1.
* **No Revisits:** The path **CANNOT** visit the same node more than once.
* **Valid Path Cells:** The path travels along the grid lines (edges between nodes). It can only occupy positions on the path (these correspond to positions with at least one even coordinate and no gap).
* **Rule Cells:** Light green cells are rule cells and have coordinates where both ‘x‘ and ‘y‘ are odd. The path goes *around* these rule cells, never *on* them. Rule cells can contain rule symbols (square, star, triangles, polyshapes) but can also be empty (no constraint).
* **Regions:** The drawn path (in combination with the grid border) divides the rule cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.

## Detailed Solving Rules
The drawn path must satisfy **ALL** applicable constraints:
1. **Path Constraints:**
* Path **MUST** start at ‘Start Node‘ (big circle on the line in the same color as the grey path{' with an ‘S‘ on it' if 'start_end_marked' in board_type else ''}) and end at ‘End Node‘ (node from which you can escape the grid to the rounded path outside{', indicated by the letter ‘E‘' if 'start_end_marked' in board_type else ''}). Both these nodes are guaranteed to be on valid path cells on the edge of the board (at least one of ‘x‘ and ‘y‘ has to be an even number).
* Path connects adjacent nodes (horizontal/vertical moves only).
* Nodes **CANNOT** be revisited.
* Path **MUST** pass through all Dots (black hexagons) that lay on the path cells.
* Path **CANNOT** pass through any Gap in the path cells (a Gap is the absence of a path at a cell which would normally be a path cell).

2. **Region-Based Rules** (Apply to areas enclosed by the path):
* **Squares:** All squares within a single region **MUST** be the same color. Squares of different colors **MUST** be separated into different regions by the path.
* **Stars:** Within a single region, each star symbol **MUST** be paired with exactly **ONE** other element (star or square) *of the same color*. Other colors within the region are irrelevant to this specific star’s rule.
* **Polyshapes (one or multiple filled squares in a specific arrangement):** The region containing this symbol **MUST** be in the specified shape (defined by the specific Polyshape arrangement). The shape must fit entirely within the region’s boundaries. If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms.
* **Negative Polyshapes (one or multiple unfilled squares in a specific arrangement):** These "subtract" shape requirements, typically within the same region as corresponding positive polyshapes. A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region. If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered ’used’ if it cancels a positive one. Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.
* **Simple Example Region:** If the a path goes through (2,0), (2,1), (2,2), (1,2), (0,2), then the rule cell (1,1) is enclosed into a region containing only itself since the grid border does the rest of the enclosure. To double check: There exist no nonvisited path cells (legal or non legal) that connect (1,1) to any other rule cell.
* **Complex Example Region:** Suppose the grid is 7x7 with rule cells at (1,1), (1,3), (1,5), (3,1), (3,3), (3,5), (5,1), (5,3), (5,5) and suppose one would want to form a region in an L-shape containing (1,1), (1,3), (3,3), then a path to separate them from the rest of the rule cells would for example be (4,0), (4,1), (4,2), (3,2), (2,2), (2,3), (2,4), (1,4), (0,4). The rest of the path cells around them at the edge don't have to be visited since they are only on the outside (x is (0 or width - 1) OR (y is (0 or height - 1)) and therefore not needed to seperate regions.

3. **Path-Based Rules (Edge Touching):**
* **Triangles (one, two, three or four):** The path **MUST** touch a specific number of edges of the cell containing the triangle symbol.
* One Triangle: Path touches **EXACTLY 1** edge of the triangle’s cell.
* Two Triangle: Path touches **EXACTLY 2** edges of the triangle’s cell.
* Three Triangle: Path touches **EXACTLY 3** edges of the triangle’s cell.
* Four Triangle: Path touches **EXACTLY 4** edges (fully surrounds) the triangle’s cell.
* Example: If a cell at (3, 3) has two triangles, the path has to go through **EXACTLY** two places of top (3, 2), right (4, 3), bottom (3, 4), or left (2, 3) to touch the rule cell.

## Task & Output Format
1. **Identifying Objects:** Analyze the grid to identify the coordinates of the Start Node, End Node, and all objects (dots and gaps on path cells; squares, stars, triangles, polyshapes on rule cells - not all types must be present in the puzzle). Keep in mind that rule cells are guaranteed to be located at coordinates where both ‘x‘ and ‘y‘ are odd. The path MUST NOT pass through these cells.
2. **Solve the Puzzle:** Determine the valid path from the Start Node to the End Node that satisfies all rules. Double check that the path doesn't go through any rule cells (where both coordinates are odd). This is the most common beginner mistake.
3. **Explain Reasoning:** Provide a step-by-step explanation of your thought process. Detail key deductions, how constraints were applied, and any backtracking or choices made.
4. **Provide Solution Path:** After the reasoning, output the exact marker string ‘####‘ followed immediately by the solution path as a list of node coordinates ‘(x, y)‘. Include all intermediate nodes from start to end (the distance between each node must be 1). 
**Example Solution Path Format:**
####
[(0, 0), (1, 0), (2, 0), (2, 1), ...]
"""

# ask lars regarding polyshapes: can they be rotated? in the prompt it says yes, in the game it wasn't in my memory
# removed: Rotation of polyshapes is generally allowed unless context implies otherwise.
# {data.get("text_visualization", "")}

# one shot example: ... is ... (objects). Since ... cells can only be located at coordinates where ..., ...