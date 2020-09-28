class Piece:
    def __init__(self, piece_type, placement: tuple):
        self.type = piece_type
        self.placement = placement

    def move(self):
        pass

    def legal_moves(self):
        pass

class Rook:
    def __init__(self):
        self.name = "R"
    
    def movement(self):


class King:
    def __init__(self):
        self.name = "K"

class Queen:
    def __init__(self):
        self.name = "Q"

class Bishop:
    def __init__(self):
        self.name = "B"

class Knight:
    def __init__(self):
        self.name = "N"

class Pawn:
    def __init__(self):
        self.name = "P"


class Position:
    pass
