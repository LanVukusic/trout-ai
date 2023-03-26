import chess

# piece types to names mapping
PIECE_TYPES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

# default description format
DEFAULT_DESCRIPTION_CAPTURE = "{capture_from_color} {piece_type} on {square_from} can capture {capture_to_color}{capture_piece_type} on {square_to}."

def generate_captures_texts(board: chess.Board, description_format: str = DEFAULT_DESCRIPTION_CAPTURE, include_pawns:bool = False) -> list:
    """Generate a list of texts describing the captures on the board.

    Parameters
    ----------
    board : chess.Board
        The board to generate the capture descriptions for.
    
    description_format : str, optional
        The format of the description. The following placeholders are available:
        - capture_from_color
        - piece_type
        - square_from
        - capture_to_color
        - capture_piece_type
        - square_to
    
    include_pawns : bool, optional
        Whether to include pawn captures in the descriptions.


    Returns
    -------
    list
        A list of strings describing the captures on the board.
    """

    descriptions = []
    legal_captures = board.generate_legal_captures()

    for move in legal_captures:
        if not include_pawns and board.piece_type_at(move.to_square) == chess.PAWN:
            continue
        try:
            descriptions.append(
                description_format.format(
                    capture_from_color="White " if board.color_at(move.from_square) == chess.WHITE else "Black ",
                    piece_type=PIECE_TYPES[board.piece_type_at(move.from_square)],
                    square_from=chess.SQUARE_NAMES[move.from_square],
                    capture_to_color="White " if board.color_at(move.to_square) == chess.WHITE else "Black ",
                    capture_piece_type=PIECE_TYPES[board.piece_type_at(move.to_square)],
                    square_to=chess.SQUARE_NAMES[move.to_square],
                )
            )
        except KeyError:
            # if the piece type is not in the mapping, skip it
            continue
    
    return descriptions


# default description format
DEFAULT_DESCRIPTION_CHECK = "{color} {piece_type} on {square} can check {check_color} king."

def generate_checks_texts(board: chess.Board, description_format: str = DEFAULT_DESCRIPTION_CHECK) -> list:
    """Generate a list of texts describing the checks on the board.

    Parameters
    ----------
    board : chess.Board
        The board to generate the check descriptions for.
    
    description_format : str, optional
        The format of the description. The following placeholders are available:
        - color
        - piece_type
        - square
        - check_color
        
    
    Returns
    -------
    list
        A list of strings describing the checks on the board.
    """

    descriptions = []
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    # find all attackers of the kings
    white_attackers = board.attackers(chess.WHITE, white_king)
    black_attackers = board.attackers(chess.BLACK, black_king)
    

    for attacker in white_attackers:
        descriptions.append(
            description_format.format(
                color="White" if board.color_at(attacker) == chess.WHITE else "Black",
                piece_type=PIECE_TYPES[board.piece_type_at(attacker)],
                square=chess.SQUARE_NAMES[attacker],
                check_color="Black",
            )
        )

    for attacker in black_attackers:
        descriptions.append(
            description_format.format(
                color="White" if board.color_at(attacker) == chess.WHITE else "Black",
                piece_type=PIECE_TYPES[board.piece_type_at(attacker)],
                square=chess.SQUARE_NAMES[attacker],
                check_color="White",
            )
        )
    
    return descriptions

# default description format for pinned pieces
DEFAULT_DESCRIPTION_PINNED = "{color} {piece_type} on {square} is pinned."

def generate_pinned_texts(board: chess.Board, description_format: str = DEFAULT_DESCRIPTION_PINNED) -> list:
    """Generate a list of texts describing the pinned pieces on the board.

    Parameters
    ----------
    board : chess.Board
        The board to generate the pinned piece descriptions for.
    
    description_format : str, optional
        The format of the description. The following placeholders are available:
        - color
        - piece_type
        - square
    
    Returns
    -------
    list
        A list of strings describing the pinned pieces on the board.
    """

    descriptions = []
    # loop over all squares and check pins with is_pinned
    white_pinned = []
    black_pinned = []

    for square in chess.SQUARES:
        if board.piece_at(square) is None:
            continue
        if board.is_pinned(chess.WHITE, square):
            white_pinned.append(square)
        if board.is_pinned(chess.BLACK, square):
            black_pinned.append(square)

    for square in white_pinned:
        descriptions.append(
            description_format.format(
                color="White",
                piece_type=PIECE_TYPES[board.piece_type_at(square)],
                square=chess.SQUARE_NAMES[square],
            )
        )

    for square in black_pinned:
        descriptions.append(
            description_format.format(
                color="Black",
                piece_type=PIECE_TYPES[board.piece_type_at(square)],
                square=chess.SQUARE_NAMES[square],
            )
        )

    return descriptions