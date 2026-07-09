

def get_prompt(data, board_type):
    # default prompt
    return f"""
You are an expert spatial detection AI specializing in detecting objects from the game ’The Witness’.
Your task is to provide an array of all objects present in the puzzle grid following the specified format.

## Grid and Object Basics
* **Grid Dimensions:** The puzzle grid has {data['grid_size']['width'] * 2 + 1} columns and {data['grid_size']['height'] * 2 + 1} rows.
* **Coordinate System:** Nodes are identified by ‘(x, y)‘ coordinates. ‘(0,0)‘ is the top-left node. ‘({data['grid_size']['width'] * 2},{data['grid_size']['height'] * 2})‘ is the bottom-right node. ‘x‘ increases to the right, ‘y‘ increases downwards. Nodes can either be path cells or rule cells.
* **Path Cells:** Path cells are dark grey and are at all positions where at least one of ‘x‘ or ‘y‘ is even. Path cells can contain dots (black hexagons) or gaps (absence of a path where there would normally be one) but can also be empty (normal path).
* **Rule Cells:** Light green cells are rule cells and have coordinates where both ‘x‘ and ‘y‘ are odd. Rule cells can contain rule symbols (square, star, triangles, polyshapes) but can also be empty.
* **Start and End Nodes:** One of the path cells on the edge of the board contains the ‘Start Node‘ (large grey circle) and another path cell on the edge contains the ‘End Node‘ (where a rounded end of the path can escape the grid).
* **Start and End Nodes:** Exactly one of the path cells on the edge of the board contains a ‘Start Node‘ (big circle on the line in the same color as the grey path{' with an ‘S‘ on it' if 'start_end_marked' in board_type else ''}) and exactly one of the path cells on the edge of the board contains an ‘End Node‘ (node from which you can escape the grid to the rounded path outside{', indicated by the letter ‘E‘' if 'start_end_marked' in board_type else ''}). Both these nodes are guaranteed to be on valid path cells on the edge of the board (at least one of ‘x‘ and ‘y‘ has to be an even number).
* **Polyshapes and Negative Polyshapes:** One or multiple filled squares (polyshapes) or unfilled squares (negative polyshapes) in a specific arrangement that defines a shape. The arrangement of the squares defines the shape.
* **Shape Representation:** Represent the shape (Y) of polyshapes and negative polyshapes as a string of 1s and 0s, where 1 represents a filled square and 0 represents an empty square and - represents a line break (from top top left to bottom right). For example, a 2x2 square would be represented as "11-11", a small L-shape could be "10-11", and a small T-shape could be "111-010", a large T-shape could be "111-010-010".
* **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black

## Symbol Legend
* `S`: **Start Node**
* `E`: **End Node**
* `+`: Empty path cell
* `N`: Empty rule cell
* `G`: **Gap** (gap where a path would normally be)
* `.`: **Dot** (black hexagon on a path cell)
* `o-X`: **Square** of color X (fills our around half of the rule cell)
* `*-X`: **Star** of color X
* `A-X`: **1 Triangle** of color X
* `B-X`: **2 Triangles** of color X
* `C-X`: **3 Triangles** of color X
* `D-X`: **4 Triangles** of color X
* `P-X-Y`: **Polyshape** (positive) of color X and shape Y (one or multiple filled small squares in a specific arrangement)
* `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape Y (one or multiple unfilled small squares in a specific arrangement)

## Task & Output Format
1. **Identifying Objects:** Analyze the grid to identify the coordinates of all objects and non-objects. Keep in mind that rule cells are guaranteed to be located at coordinates where both ‘x‘ and ‘y‘ are odd.
2. **Identify Object Colors and Shapes:** Determine the valid path from the Start Node to the End Node that satisfies all rules. Double check that the path doesn't go through any rule cells (where both coordinates are odd). This is the most common beginner mistake.
3. **Explain Reasoning:** For all coordinates, write down objects, colors, shapes and explain your reasoning if things were unclear.
4. **Provide Solution Array:** After the reasoning, output the exact marker string ‘####‘ followed immediately by the solution array as a list of list of strings (use '' to indicate a string). Include the abbreviation for all objects and all non-objects at all coordinates in the specified format. The output *MUST* follow this format to be correctly parsed.
**Example Solution Format:**
* Assume we have a 5x5 grid with a Start Node at (0, 3) and an End Node at (4, 2) and a black hexagon (dot) at (2, 0) with the rest of the path cells empty and rule cells at (1,1), (1,3), (3,1), (3,3) with a red square at (1,1), 3 green triangles at (1,3) and positive polyshapes (multiple filled squares) of color blue and small L shape at (3,3). Then the output should end with:
####
[['+', '+', '+', 'S', '+'], ['+', 'o-R', '+', 'C-G', '+'], ['.', '+', '+', '+', '+'], ['+', 'N', '+', 'P-B-11-10', '+'], ['+', '+', 'E', '+', '+']]
"""