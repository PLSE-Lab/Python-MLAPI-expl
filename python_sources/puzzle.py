"""
search the problem space of the wooden puzzle and find all solutions
"""

import csv
import logging
import math
import os
import sys
from os.path import join

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# DATA_DIRECTORY = 'data'
# for kaggle use
DATA_DIRECTORY = '../input/puzzle-data'
# TABLE_DIRECTORY = 'data'
# for kaggle use
TABLE_DIRECTORY = '.'


VALIDATION_PERIMETER_VIOLATION = 'PERIMETER_VIOLATION'
VALIDATION_OVERLAP = 'OVERLAP'
VALIDATION_VALID = 'VALID'
VALIDATION_COMPLETE = 'COMPLETE'
VALIDATION_ISOLATED_SINGLE = 'ISOLATED_SINGLE'
VALIDATION_ISOLATED_DOUBLE = 'ISOLATED_DOUBLE'
VALIDATION_ISOLATED_TRIPLE = 'ISOLATED_TRIPLE'
VALIDATION_ISOLATED_QUAD = 'ISOLATED_QUAD'
VALIDATION_ISOLATED_QUINT = 'ISOLATED_QUINT'

VALID_SET = {VALIDATION_VALID, VALIDATION_COMPLETE}

RED = (255, 0, 0, 255)
BLACK = (0, 0, 0, 255)

UNIT_VECTORS = {'k': (1, 0),
                'i': (1 / 2, math.sqrt(3) / 2),
                'u': (-1 / 2, math.sqrt(3) / 2),
                'h': (-1, 0),
                'n': (-1 / 2, -math.sqrt(3) / 2),
                'm': (1 / 2, -math.sqrt(3) / 2)}


def get_polygon_points():
    return 6


def reflect_over_x_axis(matrix) -> np.ndarray:
    """
    matrix is expected to be a Nx2 matrix where the first column is the x values
    and the second column is the y values
    """
    reflect_matrix = np.array([[1, 0], [0, -1]])
    return matrix.dot(reflect_matrix)


def cartesian_to_canvas(canvas_size, matrix: np.ndarray, pixels_per_unit=100):
    """
    pixel canvas y values move things down instead of up. This function scales
    and converts standard cartesian coordinates to the pixel canvas style of
    coordinates
    """
    centre = (canvas_size[0] / 2, -canvas_size[1] / 2)
    return reflect_over_x_axis(matrix * pixels_per_unit + centre)


def place_polygon(polygon, orientation, shift, base_angle=math.pi / 3):
    rotations_count = int(2 * math.pi / base_angle)
    flip = orientation > rotations_count - 1
    multiplier = orientation % rotations_count
    rotation_angle = base_angle * multiplier
    if flip:
        polygon = reflect_over_x_axis(polygon)
    polygon = rotate(rotation_angle, polygon)
    polygon = translation(shift[0], shift[1], polygon)
    return polygon


def rotate(theta, matrix):
    """
    theta is the angle by which to rotate the points in matrix
    matrix is expected to be a Nx2 matrix where the first column is the x values
    and the second column is the y values
    """
    r_matrix = np.array([[math.cos(theta), math.sin(theta)],
                         [-math.sin(theta), math.cos(theta)]])
    return matrix.dot(r_matrix)


def translation(x_shift, y_shift, matrix):
    """
    Shift matrix by x_shift and y_shift values
    """
    return matrix + (x_shift, y_shift)


def point_inside_polygon(x, y, poly, include_edges=True):
    """
    This is not strictly a transform, but hey it is close enough

    This code is lifted from
    https://stackoverflow.com/questions/39660851

    Test if point (x,y) is inside polygon poly.

    poly is N-vertices polygon defined as
    [(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
    (function works fine in both cases)

    Geometrical idea: point is inside polygon if horizontal beam
    to the right from point crosses polygon even number of times.
    Works fine for non-convex polygons.
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if p1y == p2y:
            if y == p1y:
                if min(p1x, p2x) <= x <= max(p1x, p2x):
                    # point is on horizontal edge
                    inside = include_edges
                    break
                elif x < min(p1x, p2x):  # point is left of the current edge
                    inside = not inside
        else:  # p1y!= p2y
            if min(p1y, p2y) <= y <= max(p1y, p2y):
                x_intersect = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

                if x == x_intersect:  # point is right on the edge
                    inside = include_edges
                    break

                if x < x_intersect:  # point is to the left from current edge
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


class GameState(object):
    def __init__(self, screen_size, data_directory='data'):
        self._screen_size = screen_size
        self._data_directory = data_directory
        self._graph_pd = None
        self._load_data()
        self._load_graph()
        self._transforms = [list() for _ in self._original_polygons]
        self._active_piece = 0

    def _load_graph(self):
        if self._graph_pd is None:
            self._graph_pd = pd.read_csv(os.path.join(self._data_directory,
                                                      'edges.csv'),
                                         index_col=0)
        self._graph = nx.from_pandas_edgelist(self._graph_pd)

    def _load_data(self):
        colour_data = pd.read_csv(os.path.join(self._data_directory,
                                               'colours.csv'))
        colours = colour_data.values[:, 1:4].tolist()
        colours = [tuple(colour) for colour in colours]
        self._colours = colours
        self._perimeter_colour = self._colours.pop()

        points_data = pd.read_csv(os.path.join(self._data_directory,
                                               'points.csv'))
        self._points = points_data.values
        self._points_canvas = cartesian_to_canvas(self._screen_size,
                                                  self._points)

        points_data = pd.read_csv(os.path.join(self._data_directory,
                                               'placement_points.csv'))
        self._placement_points = points_data.values

        perimeter_data = pd.read_csv(os.path.join(self._data_directory,
                                                  'shape9_cartesian.csv'))
        self._perimeter = cartesian_to_canvas(self._screen_size,
                                              perimeter_data.values)

        polygons = list()
        shape_csv_file_names = \
            [os.path.join(self._data_directory,
                          'shape{}_cartesian.csv'.format(i)) for i in
             range(9)]
        for in_csv in shape_csv_file_names:
            data = pd.read_csv(in_csv)
            polygons.append(data.values)
        self._original_polygons = polygons

        self._reset_state()

    def _reset_state(self):
        self._transformed_polygons = list()
        for polygon in self._original_polygons:
            self._transformed_polygons.append(np.array(polygon, copy=True))
        self._visible = [False for _ in self._original_polygons]
        self._load_graph()

    def get_points(self):
        return self._points_canvas.tolist()

    def get_placement_points(self):
        return self._placement_points.tolist()

    def get_perimeter(self):
        return self._perimeter

    def get_polygons(self):
        return [cartesian_to_canvas(self._screen_size, polygon) for
                polygon
                in self._transformed_polygons]

    def get_polygon_colours(self):
        return self._colours

    def get_perimeter_colour(self):
        return self._perimeter_colour

    def get_polygons_visible(self):
        return self._visible

    def place_polygon(self, polygon_index, placement_index, orientation):
        self._visible[polygon_index] = True
        self._transformed_polygons[polygon_index] = place_polygon(
            self._original_polygons[polygon_index], orientation,
            self._placement_points[placement_index])

    def do_action(self, action):
        if action in '123456789':
            active_piece = int(action) - 1
            if active_piece == self._active_piece and self._visible[
              active_piece]:
                self._visible[active_piece] = False
            else:
                self._active_piece = active_piece
                self._visible[active_piece] = True
        if action in UNIT_VECTORS.keys():
            shift = UNIT_VECTORS[action]
            self._transformed_polygons[self._active_piece] = translation(
                shift[0], shift[1],
                self._transformed_polygons[self._active_piece])
        if action == 'f':
            self._transformed_polygons[
                self._active_piece] = reflect_over_x_axis(
                self._transformed_polygons[self._active_piece])
        if action == 'r':
            self._transformed_polygons[self._active_piece] = rotate(
                math.pi / 3, self._transformed_polygons[self._active_piece])
        if action == 't':
            self._transformed_polygons[self._active_piece] = rotate(
                -math.pi / 3, self._transformed_polygons[self._active_piece])
        if action == 'g':
            self._reset_state()

    def get_active_piece(self):
        return self._active_piece

    def get_visible_points(self):
        visible_points = set(range(len(self._points)))
        for point in range(len(self._points)):
            for polygon, visible in enumerate(self.get_polygons_visible()):
                if visible:
                    polygon = self._transformed_polygons[polygon]
                    if point_inside_polygon(self._points[point][0],
                                            self._points[point][1],
                                            polygon):
                        visible_points.discard(point)
                        break
        return visible_points

    def get_score(self):
        return len(self._points) - len(self.get_visible_points())

    def reset(self):
        self._reset_state()

    def get_display_size(self):
        return self._screen_size

    def validate(self):
        visible_count = sum(self.get_polygons_visible())
        if visible_count * get_polygon_points() != self.get_score():
            if visible_count == 1:
                valid = VALIDATION_PERIMETER_VIOLATION
            else:
                valid = VALIDATION_OVERLAP
        else:
            if visible_count == len(self._original_polygons):
                valid = VALIDATION_COMPLETE
            else:
                valid = VALIDATION_VALID
        return valid


def draw_board(state: GameState, png_path):
    colours = state.get_polygon_colours()
    perimeter = state.get_perimeter()
    score = state.get_score()
    # noinspection PyTypeChecker
    max_score = len(state.get_points())
    score_text = '{:02d}/{:02d}'.format(score, max_score)
    font = ImageFont.load_default()
    validation_result = state.validate()
    if validation_result in VALID_SET:
        background_colour = BLACK
    else:
        background_colour = RED
    image = Image.new('RGBA', state.get_display_size(),
                      color=background_colour)
    draw_context = ImageDraw.Draw(image)
    draw_context.text((500, 40), score_text, font=font)
    draw_context.polygon(perimeter.flatten().tolist(),
                         fill=state.get_perimeter_colour())
    polygons = state.get_polygons()
    for index, polygon, visible in zip(range(len(polygons)), polygons,
                                       state.get_polygons_visible()):
        if visible:
            draw_context.polygon(polygon.flatten().tolist(),
                                 fill=colours[index])
    # noinspection PyTypeChecker
    for index, point in enumerate(state.get_points()):
        point_text = str(index)
        draw_context.text(point, point_text, font=font, fill=BLACK)
    image.save(png_path)


def draw_boards(boards):
    state = GameState((602, 522), data_directory=DATA_DIRECTORY)
    filename_format = 'board{}.png'
    for index, board in enumerate(boards):
        logging.info('drawing board: {} with settings: {}'.format(
            index,
            board
        ))
        state.reset()
        path = None
        for placement in board:
            polygon_index, point, orientation = placement
            state.place_polygon(polygon_index, point, orientation)
            path = join('images',
                        'boards',
                        state.validate(),
                        filename_format.format(
                            index))
        if path is not None:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            draw_board(state, path)


class CsvWriter(object):
    """
    Write to a csv file
    """

    def __init__(self, file_pattern,
                 pattern_fill,
                 header_fields):
        self._file_pattern = file_pattern
        self._pattern_fill = pattern_fill
        self._header_fields = header_fields
        self._file = None
        self._writer = None

    def _open_file(self):
        filename = self._file_pattern.format(*self._pattern_fill)
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self._file = open(filename, 'wt')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._header_fields)
        self._current_line = 0

    def write_row(self, row):
        if self._file is None:
            self._open_file()
        self._writer.writerow(row)


def get_coverage_dictionary():
    coverage_map = dict()
    with open('{}/coverage.csv'.format(DATA_DIRECTORY)) as f:
        reader = csv.reader(f)
        for row in reader:
            placement = tuple(map(int, row[0:3]))
            coverage = tuple(map(int, row[3:]))
            coverage_map[placement] = coverage
    return coverage_map


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

TABLE_FORMAT = '{}/validation_table_{}.csv'

COVERAGE = get_coverage_dictionary()
COVERAGE_TO_PLACEMENT = {frozenset(value): key for key, value in
                         COVERAGE.items()}
ALL_POINTS = set(range(54))
PANDAS_GRAPH = pd.read_csv(os.path.join(DATA_DIRECTORY, 'edges.csv'),
                           index_col=0)


def get_duplicates():
    duplicates = set()
    with open('{}/shape_0_duplicates.csv'.format(DATA_DIRECTORY)) as f:
        reader = csv.reader(f)
        f.readline()
        for row in reader:
            row = map(int, row)
            duplicates.add(tuple(row))
    return frozenset(duplicates)


DUPLICATES = get_duplicates()


def get_unique_single_piece_placements():
    boards = list()
    placements = set(COVERAGE_TO_PLACEMENT.values())
    for polygon_index in range(9):
        for point in range(39):
            for orientation in range(12):
                placement = (polygon_index, point, orientation)
                coverage = COVERAGE[placement]
                if len(coverage) == 6:
                    if placement not in DUPLICATES and placement in placements:
                        boards.append([placement])
    return boards


def chunk_list(l, size=3):
    it = iter(l)
    return zip(*[it] * size)


def get_fieldnames_for_table(number):
    fieldnames = list()
    for index in range(number):
        fieldnames.append('shape{}'.format(index))
        fieldnames.append('shape{}_position'.format(index))
        fieldnames.append('shape{}_orientation'.format(index))
    fieldnames.append('validation_result')
    return fieldnames


def evaluate_board(board):
    result = VALIDATION_VALID
    pieces_count = len(board)
    expected_coverage = 6 * pieces_count
    covered_points = set()
    for placement in board:
        covered_points.update(COVERAGE[placement])
    visible_points = ALL_POINTS - covered_points
    if len(covered_points) != expected_coverage:
        unexpected = True
    else:
        unexpected = False
    if unexpected:
        if pieces_count == 1:
            result = VALIDATION_PERIMETER_VIOLATION
        else:
            result = VALIDATION_OVERLAP
    else:
        graph = nx.from_pandas_edgelist(PANDAS_GRAPH)
        graph.remove_nodes_from(covered_points)
        processed_nodes = set()
        for point in visible_points:
            if point not in processed_nodes:
                nodes = set()
                for start, end in nx.dfs_edges(graph, source=point):
                    nodes.add(start)
                    nodes.add(end)
                processed_nodes.update(nodes)
                if len(nodes) == 0:
                    result = VALIDATION_ISOLATED_SINGLE
                    break
                elif len(nodes) == 2:
                    result = VALIDATION_ISOLATED_DOUBLE
                    break
                elif len(nodes) == 3:
                    result = VALIDATION_ISOLATED_TRIPLE
                    break
                elif len(nodes) == 4:
                    result = VALIDATION_ISOLATED_QUAD
                    break
                elif len(nodes) == 5:
                    result = VALIDATION_ISOLATED_QUINT
    return result


def parse_row(row):
    status = row.pop()
    placements = list()
    for item in chunk_list(map(int, row)):
        placements.append(item)
    return placements, status


def filter_single_placement_boards(boards, wanted_piece):
    filtered = list()
    logging.debug('wanted piece = {}'.format(wanted_piece))
    for board in boards:
        logging.debug(board)
        if board[0][0] == wanted_piece:
            filtered.append(board)
    return filtered


def load_boards_from_csv(filename, filter_piece=None):
    logging.info('loading file: {}'.format(filename))
    boards = list()
    with open(filename, 'rt') as f:
        # throw away the header
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            placements, status = parse_row(row)
            if status == VALIDATION_VALID:
                boards.append(placements)
    if filter_piece:
        boards = filter_single_placement_boards(boards, filter_piece)
    return boards


def iterate_boards_from_csv(filename):
    logging.info('loading file: {}'.format(filename))
    with open(filename, 'rt') as f:
        # throw away the header
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            placements, status = parse_row(row)
            if status == VALIDATION_VALID:
                yield placements


def iterate_empty_board(_):
    boards = [[]]
    for board in boards:
        yield board


def extract_piece(placement_tuple):
    return placement_tuple[0]


def get_all_possible_single_piece_placements():
    boards = list()
    size = (602, 522)
    state = GameState(size)
    placement_count = len(state.get_placement_points())
    polygon_count = len(state.get_polygons())
    del state
    for polygon_index in range(polygon_count):
        for point in range(placement_count):
            for orientation in range(12):
                boards.append([(polygon_index, point, orientation)])
    return boards


def get_placements_for_piece(piece):
    boards = list()
    for board in get_unique_single_piece_placements():
        if board[0][0] == piece and evaluate_board(board) == VALIDATION_VALID:
            boards.append(board)
    return boards


def generate_table(table_number):
    placement_piece = table_number - 1
    if table_number == 1:
        placement_boards = get_placements_for_piece(placement_piece)
        source_board_iterator = iterate_empty_board
        source_board_filename = None
    else:
        logging.debug("Loading source boards")
        source_board_iterator = iterate_boards_from_csv
        source_board_filename = \
            TABLE_FORMAT.format(TABLE_DIRECTORY,
                                table_number - 1,)
        logging.info("Loading placement boards")
        placement_boards = get_placements_for_piece(placement_piece)
    field_names = get_fieldnames_for_table(table_number)
    csv_writer = CsvWriter(file_pattern=TABLE_FORMAT,
                           pattern_fill=(TABLE_DIRECTORY,
                                         table_number,),
                           header_fields=field_names)
    logging.info('iterating source boards')
    for source_board in source_board_iterator(source_board_filename):
        pieces_on_board = set(map(extract_piece, source_board))
        for placement_board in placement_boards:
            placement_piece = placement_board[0]
            new_piece = extract_piece(placement_piece)
            if new_piece not in pieces_on_board:
                evaluation_board = list(source_board)
                evaluation_board.append(placement_piece)
                status = evaluate_board(evaluation_board)
                row = list()
                for placement in evaluation_board:
                    row.extend(placement)
                row.append(status)
                csv_writer.write_row(row)


def find_solutions():
    for table_number in range(1, 10):
        generate_table(table_number)


def draw_solutions():
    final_board_table = TABLE_FORMAT.format(TABLE_DIRECTORY, 9)
    boards = load_boards_from_csv(final_board_table)
    draw_boards(boards)


def main():
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        stream=sys.stdout)
    logging.info('start')
    find_solutions()
    draw_solutions()
    logging.info('finish')


if __name__ == '__main__':
    main()
