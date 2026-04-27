#!/usr/bin/env python
"""
Pure-Python neural evaluator trainer for the chess bot.

This script uses the evaluation ideas in index.html as a teacher:
- material
- opening / middlegame / endgame piece-square tables
- mobility
- pawn structure
- bishop pair / knight outposts / rook activity
- king pressure and shelter

It trains a small MLP without external dependencies and exports weights
to neural_weights.json. The output is designed to be a stronger evaluation
starting point than the current hand-tuned constants, while staying easy
to port back into the existing bot.

Usage:
  python train_chess_nn.py train
  python train_chess_nn.py eval "<fen>"
  python train_chess_nn.py embed
  python train_chess_nn.py refresh
  python train_chess_nn.py train-lichess --fen-file lichess_db_eval.jsonl.zst
  python train_chess_nn.py compare "<fen>" --stockfish "C:\\path\\to\\stockfish.exe"
  python train_chess_nn.py distill --stockfish "C:\\path\\to\\stockfish.exe" --fen-file positions.txt
  python train_chess_nn.py distill-refresh --stockfish "C:\\path\\to\\stockfish.exe" --fen-file positions.txt
"""

from __future__ import annotations

from tqdm import tqdm
import numpy as np
import json
import math
import os
import random
import subprocess
import sys
import time
import csv
from copy import deepcopy
from urllib.parse import urlparse
from urllib.request import urlopen
import compression.zstd as zstd
from dataclasses import dataclass
from pathlib import Path


PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}
PHASE_VALUES = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
PIECE_ORDER = ["P", "N", "B", "R", "Q", "K"]
FILES = "abcdefgh"


PST_O = {
    "P": [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 40, 40, 10, 5, 5, 0, 0, 0, 30, 30, 0, 0, 0, 5, -5, -20, 0, 0, -20, -5, 5, 5, 10, 10, -30, -30, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -50, -30, -30, -30, -30, -50, -50],
    "B": [-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -40, -10, -10, -40, -10, -20],
    "R": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 10, 15, 15, 10, 0, 0],
    "Q": [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, 20, -20, -20, -20, -20],
    "K": [-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, -20, -20, -20, -20, 20, 20, 20, 20, 50, -30, -40, -30, 50, 20],
}

PST_MG = {
    "P": [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    "B": [-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    "R": [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0],
    "Q": [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
    "K": [-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20],
}

PST_EG_KING = [-50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30, -30, -50]


KNIGHT_DELTAS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING_DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


CURATED_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",
    "r1bq1rk1/pppn1pbp/3ppnp1/8/2PPP3/2N1BN2/PP3PPP/R2QKB1R w KQ - 2 8",
    "r2q1rk1/pp2bppp/2np1n2/2p1p3/2P1P3/2NP1NP1/PP2QPBP/R1B2RK1 w - - 3 10",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 7 7",
    "r2q1rk1/1pp2ppp/p1np1n2/4p3/2P1P3/1PNP1NP1/PB2QPBP/R4RK1 w - - 2 11",
    "2rq1rk1/1b1nbppp/p2ppn2/1pp5/3PP3/1PN1BNP1/PBQ1PPBP/2RR2K1 w - - 0 13",
    "r4rk1/pp1n1ppp/2pbpn2/q1p5/2P1P3/1PNP1NPP/PBQ1NPB1/R4RK1 w - - 2 14",
    "2r2rk1/pp2qppp/2np1n2/2p1p3/2P1P3/2NP1NP1/PPQ1BPPP/2RR2K1 w - - 0 16",
    "8/2p5/3p4/3P4/2P5/8/6k1/6K1 w - - 0 1",
    "8/5pk1/4p1p1/3pP2p/3P1P1P/6P1/5K2/8 w - - 0 1",
    "6k1/5pp1/4p2p/3pP3/3P1P2/4K2P/6P1/8 w - - 0 1",
    "4rrk1/1pp2ppp/p1np1n2/8/2P1P3/1PNP1N2/PB1QBPPP/R4RK1 w - - 2 16",
    "2r2rk1/1bqnbppp/p2ppn2/1pp5/4P3/1PNPBNP1/PBQ2PBP/2RR2K1 w - - 4 14",
    "r1b2rk1/pp1n1ppp/2pbpn2/q1p5/2P1P3/1PNPBNPP/PB3PB1/R2Q1RK1 w - - 4 12",
    "r3r1k1/1bqn1ppp/p2ppn2/1pp5/3PP3/1PN1BNP1/PBQ2PBP/2RR2K1 w - - 0 15",
    "2r3k1/pp1n1ppp/2pbpn2/q1p5/2P1P3/1PNPBNPP/PB3PB1/R2Q1RK1 b - - 4 12",
    "r4rk1/pp1nqppp/2pbpn2/2p5/2P1P3/1PNPBNPP/PB1Q1PB1/R4RK1 b - - 0 14",
    "5rk1/1p3ppp/p2b1n2/3P4/2P5/1P3NP1/P4PBP/5RK1 w - - 0 24",
    "8/8/2k5/2p5/2P5/3K4/8/8 w - - 0 1",
]

DEFAULT_STOCKFISH_DEPTH = 12
DEFAULT_DISTILL_EPOCHS = 260
DEFAULT_STOCKFISH_BLEND = 0.8
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_RANDOM_SEED = 7


@dataclass
class Position:
    board: list[str | None]
    turn: str
    castling: str
    ep: str


@dataclass
class StockfishSample:
    fen: str
    teacher: float
    stockfish: float
    blended: float


def square_name(index: int) -> str:
    return FILES[index % 8] + str(index // 8 + 1)


def file_of(index: int) -> int:
    return index % 8


def rank_of(index: int) -> int:
    return index // 8


def on_board(file_idx: int, rank_idx: int) -> bool:
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


def index_of(file_idx: int, rank_idx: int) -> int:
    return rank_idx * 8 + file_idx


def parse_fen(fen: str) -> Position:
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN: {fen}")
    board = [None] * 64
    rank_idx = 7
    file_idx = 0
    for ch in parts[0]:
        if ch == "/":
            rank_idx -= 1
            file_idx = 0
            continue
        if ch.isdigit():
            file_idx += int(ch)
            continue
        board[index_of(file_idx, rank_idx)] = ch
        file_idx += 1
    return Position(board=board, turn=parts[1], castling=parts[2], ep=parts[3])


def is_valid_fen(fen: str) -> bool:
    try:
        parse_fen(fen)
        return True
    except Exception:
        return False


def mirrored_index_for_white(index: int, piece: str) -> int:
    return 63 - index if piece.isupper() else index


def iter_pieces(board: list[str | None], piece: str) -> list[int]:
    return [idx for idx, p in enumerate(board) if p == piece]


def attack_squares(board: list[str | None], index: int, piece: str) -> set[int]:
    file_idx = file_of(index)
    rank_idx = rank_of(index)
    side_white = piece.isupper()
    upper = piece.upper()
    attacks: set[int] = set()

    if upper == "P":
        step = 1 if side_white else -1
        for df in (-1, 1):
            nf = file_idx + df
            nr = rank_idx + step
            if on_board(nf, nr):
                attacks.add(index_of(nf, nr))
        return attacks

    if upper == "N":
        for df, dr in KNIGHT_DELTAS:
            nf = file_idx + df
            nr = rank_idx + dr
            if on_board(nf, nr):
                attacks.add(index_of(nf, nr))
        return attacks

    if upper == "K":
        for df, dr in KING_DELTAS:
            nf = file_idx + df
            nr = rank_idx + dr
            if on_board(nf, nr):
                attacks.add(index_of(nf, nr))
        return attacks

    directions = []
    if upper in ("B", "Q"):
        directions.extend(DIAG)
    if upper in ("R", "Q"):
        directions.extend(ORTHO)

    for df, dr in directions:
        nf = file_idx + df
        nr = rank_idx + dr
        while on_board(nf, nr):
            sq = index_of(nf, nr)
            attacks.add(sq)
            if board[sq] is not None:
                break
            nf += df
            nr += dr
    return attacks


def attacked_by(board: list[str | None], side: str) -> set[int]:
    squares: set[int] = set()
    for idx, piece in enumerate(board):
        if piece is None:
            continue
        if side == "w" and piece.isupper():
            squares |= attack_squares(board, idx, piece)
        elif side == "b" and piece.islower():
            squares |= attack_squares(board, idx, piece)
    return squares


def piece_mobility(board: list[str | None], index: int, piece: str, enemy_pawn_zone: set[int]) -> int:
    own_white = piece.isupper()
    moves = 0
    for sq in attack_squares(board, index, piece):
        occupant = board[sq]
        if occupant is not None and occupant.isupper() == own_white:
            continue
        if sq in enemy_pawn_zone:
            continue
        moves += 1
    return moves


def king_ring(index: int) -> set[int]:
    ring = {index}
    file_idx = file_of(index)
    rank_idx = rank_of(index)
    for df, dr in KING_DELTAS:
        nf = file_idx + df
        nr = rank_idx + dr
        if on_board(nf, nr):
            ring.add(index_of(nf, nr))
    return ring


def file_counts(board: list[str | None], pawn: str) -> list[int]:
    counts = [0] * 8
    for idx, piece in enumerate(board):
        if piece == pawn:
            counts[file_of(idx)] += 1
    return counts


def is_passed_pawn(board: list[str | None], index: int, piece: str) -> bool:
    file_idx = file_of(index)
    rank_idx = rank_of(index)
    enemy = "p" if piece.isupper() else "P"
    direction = 1 if piece.isupper() else -1
    for df in (-1, 0, 1):
        nf = file_idx + df
        if not 0 <= nf < 8:
            continue
        nr = rank_idx + direction
        while 0 <= nr < 8:
            sq = index_of(nf, nr)
            if board[sq] == enemy:
                return False
            nr += direction
    return True


def evaluate_teacher(fen: str) -> tuple[list[float], float]:
    pos = parse_fen(fen)
    board = pos.board
    white_attacks = attacked_by(board, "w")
    black_attacks = attacked_by(board, "b")

    opening = 0.0
    middlegame = 0.0
    endgame = 0.0
    phase = 0
    material_terms = []
    counts = {}

    for piece_code in PIECE_ORDER:
        white_count = 0
        black_count = 0
        for idx, piece in enumerate(board):
            if piece == piece_code:
                white_count += 1
                mirrored = mirrored_index_for_white(idx, piece)
                opening += PIECE_VALUES[piece_code] + PST_O[piece_code][mirrored]
                middlegame += PIECE_VALUES[piece_code] + PST_MG[piece_code][mirrored]
                endgame += PIECE_VALUES[piece_code] + (PST_EG_KING[mirrored] if piece_code == "K" else PST_MG[piece_code][mirrored])
                phase += PHASE_VALUES[piece_code]
            elif piece == piece_code.lower():
                black_count += 1
                mirrored = mirrored_index_for_white(idx, piece)
                opening -= PIECE_VALUES[piece_code] + PST_O[piece_code][mirrored]
                middlegame -= PIECE_VALUES[piece_code] + PST_MG[piece_code][mirrored]
                endgame -= PIECE_VALUES[piece_code] + (PST_EG_KING[mirrored] if piece_code == "K" else PST_MG[piece_code][mirrored])
                phase += PHASE_VALUES[piece_code]
        material_terms.append(float(white_count - black_count))
        counts[piece_code] = white_count
        counts[piece_code.lower()] = black_count

    phase = min(phase, 24)
    opening_weight = max(0.0, (phase - 20) / 4.0)
    tapered = (middlegame * phase + endgame * (24 - phase)) / 24.0
    score = opening * opening_weight + tapered * (1.0 - opening_weight)
    score += 10.0

    threat_score = 0.0
    attacked_white = defended_white = attacked_black = defended_black = 0.0
    for idx, piece in enumerate(board):
        if piece is None:
            continue
        value = PIECE_VALUES[piece.upper()]
        if piece.isupper():
            if idx in black_attacks:
                attacked_white += 1.0
                if idx in white_attacks:
                    defended_white += 1.0
                    threat_score -= value * 0.15
                else:
                    threat_score -= value * 0.5
        else:
            if idx in white_attacks:
                attacked_black += 1.0
                if idx in black_attacks:
                    defended_black += 1.0
                    threat_score += value * 0.15
                else:
                    threat_score += value * 0.5
    score += threat_score

    white_pawn_zone = black_attacks
    black_pawn_zone = white_attacks
    mobility_weights = {"N": 4, "B": 4, "R": 2, "Q": 1}
    mobility = 0.0
    white_mobility = 0.0
    black_mobility = 0.0
    for idx, piece in enumerate(board):
        if piece is None or piece.upper() not in mobility_weights:
            continue
        mob = piece_mobility(board, idx, piece, white_pawn_zone if piece.isupper() else black_pawn_zone)
        weighted = mob * mobility_weights[piece.upper()]
        if piece.isupper():
            mobility += weighted
            white_mobility += weighted
        else:
            mobility -= weighted
            black_mobility += weighted
    score += mobility

    doubled_white = doubled_black = isolated_white = isolated_black = 0.0
    passed_white = passed_black = 0.0
    white_files = file_counts(board, "P")
    black_files = file_counts(board, "p")
    for f in range(8):
        if white_files[f] > 1:
            doubled_white += white_files[f] - 1
            score -= (white_files[f] - 1) * 18.0
        if black_files[f] > 1:
            doubled_black += black_files[f] - 1
            score += (black_files[f] - 1) * 18.0

        if white_files[f] > 0:
            left = white_files[f - 1] if f > 0 else 0
            right = white_files[f + 1] if f < 7 else 0
            if left + right == 0:
                isolated_white += white_files[f]
                score -= white_files[f] * 15.0
        if black_files[f] > 0:
            left = black_files[f - 1] if f > 0 else 0
            right = black_files[f + 1] if f < 7 else 0
            if left + right == 0:
                isolated_black += black_files[f]
                score += black_files[f] * 15.0

    for idx, piece in enumerate(board):
        if piece == "P" and is_passed_pawn(board, idx, piece):
            bonus = 25 + rank_of(idx) * 10
            scaled = bonus * (1.0 + (24 - phase) / 24.0)
            passed_white += 1.0
            score += scaled
        elif piece == "p" and is_passed_pawn(board, idx, piece):
            bonus = 25 + (7 - rank_of(idx)) * 10
            scaled = bonus * (1.0 + (24 - phase) / 24.0)
            passed_black += 1.0
            score -= scaled

    bishop_pair = 0.0
    if counts["B"] >= 2:
        score += 35.0
        bishop_pair += 1.0
    if counts["b"] >= 2:
        score -= 35.0
        bishop_pair -= 1.0

    outpost = 0.0
    for idx, piece in enumerate(board):
        if piece == "N":
            r = rank_of(idx)
            if 3 <= r <= 5:
                defended = any(board[sq] == "P" for sq in attack_squares(board, idx, "p"))
                enemy_can_chase = any(board[sq] == "p" for sq in attack_squares(board, idx, "P"))
                if defended and not enemy_can_chase:
                    outpost += 1.0
                    score += 30.0
        elif piece == "n":
            r = rank_of(idx)
            if 2 <= r <= 4:
                defended = any(board[sq] == "p" for sq in attack_squares(board, idx, "P"))
                enemy_can_chase = any(board[sq] == "P" for sq in attack_squares(board, idx, "p"))
                if defended and not enemy_can_chase:
                    outpost -= 1.0
                    score -= 30.0

    rook_7th = open_rooks = semi_open_rooks = 0.0
    for idx, piece in enumerate(board):
        if piece == "R":
            if rank_of(idx) == 6:
                rook_7th += 1.0
                score += 25.0
            friendly = white_files[file_of(idx)]
            enemy = black_files[file_of(idx)]
            if friendly == 0:
                semi_open_rooks += 1.0
                score += 15.0
                if enemy == 0:
                    open_rooks += 1.0
                    score += 10.0
        elif piece == "r":
            if rank_of(idx) == 1:
                rook_7th -= 1.0
                score -= 25.0
            friendly = black_files[file_of(idx)]
            enemy = white_files[file_of(idx)]
            if friendly == 0:
                semi_open_rooks -= 1.0
                score -= 15.0
                if enemy == 0:
                    open_rooks -= 1.0
                    score -= 10.0

    white_king = next((idx for idx, p in enumerate(board) if p == "K"), None)
    black_king = next((idx for idx, p in enumerate(board) if p == "k"), None)
    king_danger = king_shelter = 0.0
    queen_pressure = minor_pressure = 0.0
    if white_king is not None:
        ring = king_ring(white_king)
        danger = sum(1 for sq in ring if sq in black_attacks)
        king_danger -= danger
        score -= danger * 14.0
        white_file = file_of(white_king)
        shelter_sq = white_king + 7 if white_file > 4 else white_king + 9
        if 0 <= shelter_sq < 64 and board[shelter_sq] == "P":
            king_shelter += 1.0
            score += 15.0
        else:
            king_shelter -= 1.0
            score -= 20.0
        if any(piece == "q" and ring & attack_squares(board, idx, piece) for idx, piece in enumerate(board) if piece == "q"):
            queen_pressure -= 1.0
            score -= 18.0
        if any(piece in ("n", "b") and ring & attack_squares(board, idx, piece) for idx, piece in enumerate(board) if piece in ("n", "b")):
            minor_pressure -= 1.0
            score -= 10.0

    if black_king is not None:
        ring = king_ring(black_king)
        danger = sum(1 for sq in ring if sq in white_attacks)
        king_danger += danger
        score += danger * 14.0
        black_file = file_of(black_king)
        shelter_sq = black_king - 9 if black_file > 4 else black_king - 7
        if 0 <= shelter_sq < 64 and board[shelter_sq] == "p":
            king_shelter -= 1.0
            score -= 15.0
        else:
            king_shelter += 1.0
            score += 20.0
        if any(piece == "Q" and ring & attack_squares(board, idx, piece) for idx, piece in enumerate(board) if piece == "Q"):
            queen_pressure += 1.0
            score += 18.0
        if any(piece in ("N", "B") and ring & attack_squares(board, idx, piece) for idx, piece in enumerate(board) if piece in ("N", "B")):
            minor_pressure += 1.0
            score += 10.0

    white_center = sum(1 for sq in (27, 28, 35, 36) if sq in white_attacks)
    black_center = sum(1 for sq in (27, 28, 35, 36) if sq in black_attacks)
    center_control = float(white_center - black_center)
    score += center_control * 6.0

    development = 0.0
    if board[1] != "N":
        development += 1.0
    if board[6] != "N":
        development += 1.0
    if board[2] != "B":
        development += 1.0
    if board[5] != "B":
        development += 1.0
    if board[57] != "n":
        development -= 1.0
    if board[62] != "n":
        development -= 1.0
    if board[58] != "b":
        development -= 1.0
    if board[61] != "b":
        development -= 1.0
    score += development * 8.0

    features = [
        phase / 24.0,
        opening / 20000.0,
        middlegame / 20000.0,
        endgame / 20000.0,
        threat_score / 3000.0,
        mobility / 200.0,
        center_control / 8.0,
        development / 8.0,
        king_danger / 16.0,
        king_shelter / 4.0,
        queen_pressure / 4.0,
        minor_pressure / 4.0,
        bishop_pair,
        outpost / 4.0,
        rook_7th / 4.0,
        semi_open_rooks / 4.0,
        open_rooks / 4.0,
        doubled_white / 8.0,
        doubled_black / 8.0,
        isolated_white / 8.0,
        isolated_black / 8.0,
        passed_white / 8.0,
        passed_black / 8.0,
        attacked_white / 16.0,
        defended_white / 16.0,
        attacked_black / 16.0,
        defended_black / 16.0,
        white_mobility / 200.0,
        black_mobility / 200.0,
        1.0 if pos.turn == "w" else -1.0,
    ]
    features.extend(material_terms)
    return features, score / 1000.0


def mirror_fen(fen: str) -> str:
    pos = parse_fen(fen)
    mirrored = [None] * 64
    for idx, piece in enumerate(pos.board):
        if piece is None:
            continue
        file_idx = file_of(idx)
        rank_idx = rank_of(idx)
        new_idx = index_of(7 - file_idx, 7 - rank_idx)
        mirrored[new_idx] = piece.swapcase()
    rows = []
    for rank_idx in range(7, -1, -1):
        run = 0
        row = []
        for file_idx in range(8):
            piece = mirrored[index_of(file_idx, rank_idx)]
            if piece is None:
                run += 1
            else:
                if run:
                    row.append(str(run))
                    run = 0
                row.append(piece)
        if run:
            row.append(str(run))
        rows.append("".join(row))
    turn = "b" if pos.turn == "w" else "w"
    castling = pos.castling.translate(str.maketrans("KQkq", "kqKQ"))
    if not castling:
        castling = "-"
    ep = "-"
    return "/".join(rows) + f" {turn} {castling} {ep} 0 1"


def build_dataset() -> list[tuple[list[float], float]]:
    dataset = []
    seen = set()
    for fen in CURATED_FENS:
        for variant in (fen, mirror_fen(fen)):
            if variant in seen:
                continue
            seen.add(variant)
            features, label = evaluate_teacher(variant)
            dataset.append((features, label))
    return dataset


def build_feature_dataset(fens: list[str]) -> list[tuple[str, list[float], float]]:
    dataset = []
    seen = set()
    for fen in fens:
        for variant in (fen, mirror_fen(fen)):
            if variant in seen:
                continue
            seen.add(variant)
            features, label = evaluate_teacher(variant)
            dataset.append((variant, features, label))
    return dataset


class StockfishEngine:
    def __init__(self, exe_path: str, threads: int = 1, hash_mb: int = 32) -> None:
        self.exe_path = exe_path
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok", 10.0)
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._wait_for("readyok", 10.0)

    def _send(self, line: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("Stockfish stdin unavailable")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, marker: str, timeout_s: float) -> str:
        deadline = time.time() + timeout_s
        lines = []
        if self.proc.stdout is None:
            raise RuntimeError("Stockfish stdout unavailable")
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            lines.append(line)
            if marker in line:
                return line
        raise TimeoutError(f"Timed out waiting for {marker!r}. Last output: {lines[-5:]}")

    def evaluate_fen(
        self,
        fen: str,
        depth: int = DEFAULT_STOCKFISH_DEPTH,
        timeout_s: float = 8.0,
    ) -> float:
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send("isready")
        self._wait_for("readyok", 10.0)
        self._send(f"go depth {depth}")

        best_score: float | None = None
        deadline = time.time() + timeout_s
        if self.proc.stdout is None:
            raise RuntimeError("Stockfish stdout unavailable")
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith("info ") and " score " in line:
                parsed = parse_uci_score(line)
                if parsed is not None:
                    best_score = parsed
            elif line.startswith("bestmove"):
                return 0.0 if best_score is None else best_score
        raise TimeoutError(f"Stockfish timed out on FEN: {fen}")

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def parse_uci_score(line: str) -> float | None:
    tokens = line.split()
    for idx, token in enumerate(tokens):
        if token != "score" or idx + 2 >= len(tokens):
            continue
        kind = tokens[idx + 1]
        raw = tokens[idx + 2]
        if kind == "cp":
            return int(raw) / 100.0
        if kind == "mate":
            mate_in = int(raw)
            sign = 1.0 if mate_in > 0 else -1.0
            return sign * (100.0 - min(99.0, abs(mate_in)))
    return None


def maybe_extract_fen_from_text(text: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    if is_valid_fen(candidate):
        return candidate
    return None


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def parse_fens_from_text(name: str, text: str) -> list[str]:
    suffix = Path(name).suffix.lower()
    fens: list[str] = []

    if suffix in {".json", ".jsonl"}:
        if suffix == ".json":
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        fen = maybe_extract_fen_from_text(item)
                        if fen:
                            fens.append(fen)
                    elif isinstance(item, dict):
                        for key in ("fen", "position", "epd"):
                            if key in item and isinstance(item[key], str):
                                fen = maybe_extract_fen_from_text(item[key])
                                if fen:
                                    fens.append(fen)
                                    break
            elif isinstance(data, dict):
                for key in ("fens", "positions"):
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            if isinstance(item, str):
                                fen = maybe_extract_fen_from_text(item)
                                if fen:
                                    fens.append(fen)
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = line
                if isinstance(item, str):
                    fen = maybe_extract_fen_from_text(item)
                    if fen:
                        fens.append(fen)
                elif isinstance(item, dict):
                    for key in ("fen", "position", "epd"):
                        if key in item and isinstance(item[key], str):
                            fen = maybe_extract_fen_from_text(item[key])
                            if fen:
                                fens.append(fen)
                                break
        return fens

    if suffix == ".csv":
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            for key in ("fen", "position", "epd", "FEN"):
                if key in row and row[key]:
                    fen = maybe_extract_fen_from_text(row[key])
                    if fen:
                        fens.append(fen)
                        break
        return fens

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen = maybe_extract_fen_from_text(line)
        if fen:
            fens.append(fen)
            continue
        if "," in line:
            for part in line.split(","):
                fen = maybe_extract_fen_from_text(part)
                if fen:
                    fens.append(fen)
                    break
    return fens


def reservoir_sample(items, sample_size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    for idx, item in enumerate(items, start=1):
        if len(sample) < sample_size:
            sample.append(item)
            continue
        slot = rng.randrange(idx)
        if slot < sample_size:
            sample[slot] = item
    return sample


def iter_fens_from_jsonl_stream(stream):
    for raw_line in stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            item = line
        if isinstance(item, dict):
            for key in ("fen", "position", "epd"):
                if key in item and isinstance(item[key], str):
                    fen = maybe_extract_fen_from_text(item[key])
                    if fen:
                        yield fen
                        break
        elif isinstance(item, str):
            fen = maybe_extract_fen_from_text(item)
            if fen:
                yield fen


def score_from_eval_payload(item: dict) -> float | None:
    evals = item.get("evals")
    if not isinstance(evals, list) or not evals:
        return None
    best = max((e for e in evals if isinstance(e, dict)), key=lambda e: e.get("depth", 0), default=None)
    if not best:
        return None
    pvs = best.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        return None
    pv0 = pvs[0]
    if not isinstance(pv0, dict):
        return None
    if "cp" in pv0:
        return float(pv0["cp"]) / 100.0
    if "mate" in pv0:
        mate_in = int(pv0["mate"])
        sign = 1.0 if mate_in > 0 else -1.0
        return sign * (100.0 - min(99.0, abs(mate_in)))
    return None


def iter_labeled_fens_from_jsonl_stream(stream):
    for raw_line in stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        fen = None
        for key in ("fen", "position", "epd"):
            if key in item and isinstance(item[key], str):
                fen = maybe_extract_fen_from_text(item[key])
                if fen:
                    break
        if not fen:
            continue
        score = score_from_eval_payload(item)
        if score is None:
            continue
        yield fen, score


def load_fens_from_file(path: Path) -> list[str]:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes[-2:] == [".jsonl", ".zst"]:
        with zstd.open(path, mode="rt", encoding="utf-8") as handle:
            return list(iter_fens_from_jsonl_stream(handle))
    if suffixes[-1:] == [".zst"]:
        with zstd.open(path, mode="rt", encoding="utf-8") as handle:
            return parse_fens_from_text(path.stem, handle.read())
    return parse_fens_from_text(path.name, path.read_text(encoding="utf-8"))


def import_fens(
    source: str | None,
    sample_size: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[str]:
    if not source:
        return []
    if is_url(source):
        parsed = urlparse(source)
        name = Path(parsed.path).name or "remote.jsonl"
        if name.endswith(".jsonl.zst"):
            with urlopen(source) as response:
                with zstd.open(response, mode="rt", encoding="utf-8") as handle:
                    stream = iter_fens_from_jsonl_stream(handle)
                    fens = reservoir_sample(stream, sample_size, seed) if sample_size else list(stream)
            return fens
        with urlopen(source) as response:
            text = response.read().decode("utf-8", errors="replace")
        fens = parse_fens_from_text(name, text)
        rng = random.Random(seed)
        rng.shuffle(fens)
        return fens[:sample_size] if sample_size is not None else fens

    path = Path(source)
    files: list[Path] = []
    if path.is_dir():
        for pattern in ("*.fen", "*.txt", "*.epd", "*.json", "*.jsonl", "*.csv"):
            files.extend(sorted(path.rglob(pattern)))
        for pattern in ("*.jsonl.zst", "*.zst"):
            files.extend(sorted(path.rglob(pattern)))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"FEN source not found: {source}")

    all_fens: list[str] = []
    seen = set()
    for file in files:
        if sample_size is not None and file.name.endswith(".jsonl.zst"):
            with zstd.open(file, mode="rt", encoding="utf-8") as handle:
                current_fens = reservoir_sample(iter_fens_from_jsonl_stream(handle), sample_size, seed)
        else:
            current_fens = load_fens_from_file(file)
        for fen in current_fens:
            if fen not in seen:
                seen.add(fen)
                all_fens.append(fen)

    rng = random.Random(seed)
    rng.shuffle(all_fens)
    if sample_size is not None:
        all_fens = all_fens[:sample_size]
    return all_fens


def import_labeled_lichess_positions(
    source: str,
    sample_size: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[tuple[str, float]]:
    if is_url(source):
        parsed = urlparse(source)
        name = Path(parsed.path).name or "remote.jsonl"
        if name.endswith(".jsonl.zst"):
            with urlopen(source) as response:
                with zstd.open(response, mode="rt", encoding="utf-8") as handle:
                    rows = iter_labeled_fens_from_jsonl_stream(handle)
                    return reservoir_sample(rows, sample_size, seed) if sample_size else list(rows)
        with urlopen(source) as response:
            text = response.read().decode("utf-8", errors="replace")
        rows = []
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            fen = maybe_extract_fen_from_text(item.get("fen", ""))
            score = score_from_eval_payload(item)
            if fen and score is not None:
                rows.append((fen, score))
        rng = random.Random(seed)
        rng.shuffle(rows)
        return rows[:sample_size] if sample_size is not None else rows

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Lichess source not found: {source}")
    if path.name.endswith(".jsonl.zst"):
        with zstd.open(path, mode="rt", encoding="utf-8") as handle:
            rows = iter_labeled_fens_from_jsonl_stream(handle)
            return reservoir_sample(rows, sample_size, seed) if sample_size else list(rows)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        fen = maybe_extract_fen_from_text(item.get("fen", ""))
        score = score_from_eval_payload(item)
        if fen and score is not None:
            rows.append((fen, score))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:sample_size] if sample_size is not None else rows


def build_dataset_from_labeled_positions(
    labeled_positions: list[tuple[str, float]],
) -> list[tuple[list[float], float]]:
    dataset = []
    for fen, label in labeled_positions:
        features, _ = evaluate_teacher(fen)
        dataset.append((features, label))
    return dataset


def resolve_stockfish_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        return env_path
    raise FileNotFoundError(
        "Stockfish path not provided. Pass --stockfish <path> or set STOCKFISH_PATH."
    )


def build_stockfish_supervised_dataset(
    stockfish_path: str,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    blend: float = DEFAULT_STOCKFISH_BLEND,
    extra_fens: list[str] | None = None,
    limit: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    use_curated: bool = True,
    stockfish_timeout_s: float = 8.0,
) -> tuple[list[tuple[list[float], float]], list[StockfishSample]]:
    fens = CURATED_FENS[:] if use_curated else []
    if extra_fens:
        fens.extend(extra_fens)
    
    feature_rows = build_feature_dataset(fens)
    rng = random.Random(seed)
    rng.shuffle(feature_rows)
    if limit is not None:
        feature_rows = feature_rows[:limit]
    
    samples: list[StockfishSample] = []
    dataset: list[tuple[list[float], float]] = []

    print(f"Analyzing {len(feature_rows)} positions with Stockfish...")
    
    # Progress bar for Stockfish evaluation
    with StockfishEngine(stockfish_path) as engine:
        for fen, features, teacher_label in tqdm(feature_rows, desc="Stockfish Eval"):
            try:
                sf_score = engine.evaluate_fen(fen, depth=depth, timeout_s=stockfish_timeout_s)
            except Exception:
                continue
                
            blended = teacher_label * (1.0 - blend) + sf_score * blend
            samples.append(StockfishSample(fen, teacher_label, sf_score, blended))
            dataset.append((features, blended))
            
    return dataset, samples


import numpy as np

class MLP:
    def __init__(self, input_size: int, hidden1: int = 128, hidden2: int = 64, seed: int = 7) -> None:
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # Initialize weights as NumPy arrays
        self.w1 = np.random.uniform(-0.2, 0.2, (hidden1, input_size))
        self.b1 = np.zeros(hidden1)
        self.w2 = np.random.uniform(-0.2, 0.2, (hidden2, hidden1))
        self.b2 = np.zeros(hidden2)
        self.w3 = np.random.uniform(-0.2, 0.2, hidden2)
        self.b3 = 0.0

    def forward(self, x: list[float]) -> tuple[np.ndarray, np.ndarray, float]:
        # Convert input list to numpy array
        x_arr = np.array(x)
        
        # Layer 1: Dot product + Tanh
        h1 = np.tanh(np.dot(self.w1, x_arr) + self.b1)
        
        # Layer 2: Dot product + Tanh
        h2 = np.tanh(np.dot(self.w2, h1) + self.b2)
        
        # Output: Dot product
        out = float(np.dot(self.w3, h2) + self.b3)
        
        return h1, h2, out

    def predict(self, x: list[float]) -> float:
        _, _, out = self.forward(x)
        return out

    def state_dict(self) -> dict:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3,
        }

    def load_state_dict(self, state: dict) -> None:
        self.w1 = np.array(state["w1"])
        self.b1 = np.array(state["b1"])
        self.w2 = np.array(state["w2"])
        self.b2 = np.array(state["b2"])
        self.w3 = np.array(state["w3"])
        self.b3 = state["b3"]

    def train_epoch(self, samples: list[tuple[list[float], float]], lr: float) -> float:
        random.shuffle(samples)
        total_loss = 0.0
        
        for x, y in samples:
            x_arr = np.array(x)
            h1, h2, pred = self.forward(x)
            err = pred - y
            total_loss += err * err

            # --- Backpropagation (Vectorized) ---
            d_out = 2.0 * err
            
            # Gradients for Layer 3
            grad_w3 = d_out * h2
            grad_b3 = d_out

            # Gradients for Layer 2
            # tanh_prime(h2) is (1 - h2^2)
            d_h2 = (d_out * self.w3) * (1.0 - h2**2)
            grad_w2 = np.outer(d_h2, h1)
            grad_b2 = d_h2

            # Gradients for Layer 1
            d_h1 = np.dot(self.w2.T, d_h2) * (1.0 - h1**2)
            grad_w1 = np.outer(d_h1, x_arr)
            grad_b1 = d_h1

            # Update weights
            self.w3 -= lr * grad_w3
            self.b3 -= lr * grad_b3
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1

        return total_loss / max(1, len(samples))

    def export(self, feature_mean: list[float], feature_std: list[float], path: Path) -> None:
        payload = {
            "architecture": {"input": self.input_size, "hidden1": self.hidden1, "hidden2": self.hidden2, "output": 1},
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


def compute_normalization(dataset: list[tuple[list[float], float]]) -> tuple[list[float], list[float]]:
    width = len(dataset[0][0])
    mean = [0.0] * width
    std = [0.0] * width

    for x, _ in dataset:
        for i, value in enumerate(x):
            mean[i] += value
    mean = [v / len(dataset) for v in mean]

    for x, _ in dataset:
        for i, value in enumerate(x):
            std[i] += (value - mean[i]) ** 2
    std = [math.sqrt(v / len(dataset)) if v > 1e-12 else 1.0 for v in std]
    return mean, std


def apply_normalization(
    dataset: list[tuple[list[float], float]],
    mean: list[float],
    std: list[float],
) -> list[tuple[list[float], float]]:
    width = len(dataset[0][0])
    normalized = []
    for x, y in dataset:
        normalized.append(([(x[i] - mean[i]) / std[i] for i in range(width)], y))
    return normalized


def normalize_dataset(dataset: list[tuple[list[float], float]]) -> tuple[list[tuple[list[float], float]], list[float], list[float]]:
    mean, std = compute_normalization(dataset)
    return apply_normalization(dataset, mean, std), mean, std


def split_dataset(
    dataset: list[tuple[list[float], float]],
    validation_split: float,
    seed: int,
) -> tuple[list[tuple[list[float], float]], list[tuple[list[float], float]]]:
    rows = dataset[:]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < 2 or validation_split <= 0:
        return rows, []
    val_size = max(1, int(len(rows) * validation_split))
    if val_size >= len(rows):
        val_size = len(rows) - 1
    return rows[val_size:], rows[:val_size]


def evaluate_loss(model: MLP, samples: list[tuple[list[float], float]]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mse = 0.0
    mae = 0.0
    for x, y in samples:
        pred = model.predict(x)
        err = pred - y
        mse += err * err
        mae += abs(err)
    return mse / len(samples), mae / len(samples)


def train(
    dataset: list[tuple[list[float], float]] | None = None,
    epochs: int = 220,
    base_lr: float = 0.005,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    # 1. Prepare Data
    dataset = dataset or build_dataset()
    train_rows, val_rows = split_dataset(dataset, validation_split, seed)
    mean, std = compute_normalization(train_rows)
    train_norm = apply_normalization(train_rows, mean, std)
    val_norm = apply_normalization(val_rows, mean, std) if val_rows else []
    
    # 2. Initialize Model (Make sure you use the NumPy version of MLP provided earlier)
    model = MLP(input_size=len(train_norm[0][0]), seed=seed)
    best_state = model.state_dict()
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\nStarting Training (Size: {len(train_norm)} train, {len(val_norm)} val)")
    
    # 3. Training Loop with Progress Bar
    # desc: Label on the left, unit: what we are counting
    pbar = tqdm(range(epochs), desc="Training NN", unit="epoch")
    
    for epoch in pbar:
        # Learning rate decay
        lr = base_lr * (0.998 ** epoch)
        
        # Train one epoch (NumPy version is much faster)
        train_loss = model.train_epoch(train_norm, lr)
        
        # Validation
        val_loss, val_mae = evaluate_loss(model, val_norm)
        
        # Save best model
        if val_norm and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = model.state_dict()
        
        # Update loader text every epoch
        if val_norm:
            pbar.set_postfix({
                "t_loss": f"{train_loss:.4f}", 
                "v_mae": f"{val_mae:.4f}"
            })
        else:
            pbar.set_postfix({"t_loss": f"{train_loss:.4f}"})

    # 4. Finalize
    if val_norm:
        model.load_state_dict(best_state)
        final_val_loss, final_val_mae = evaluate_loss(model, val_norm)
        print(f"\nTraining Complete!")
        print(f" > Best Epoch: {best_epoch:03d}")
        print(f" > Final Val Loss: {final_val_loss:.6f}")
        print(f" > Final Val MAE:  {final_val_mae:.6f}")

    # 5. Export
    weights_path = Path(__file__).with_name("neural_weights.json")
    model.export(mean, std, weights_path)
    print(f"Saved weights to {weights_path}")

    # 6. Sanity Check Samples
    sample_fens = [
        CURATED_FENS[0],
        CURATED_FENS[1],
        CURATED_FENS[9],
    ]
    print("\n--- Model Predictions (Sanity Check) ---")
    for fen in sample_fens:
        print(f"\nFEN: {fen}")
        print(f"  Teacher Eval: {evaluate_teacher(fen)[1]:.3f}")
        print(f"  Network Eval: {evaluate_loaded(fen, weights_path):.3f}")


def compare_with_stockfish(
    fen: str,
    stockfish_path: str,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
) -> None:
    teacher = evaluate_teacher(fen)[1]
    weights_path = Path(__file__).with_name("neural_weights.json")
    network = evaluate_loaded(fen, weights_path) if weights_path.exists() else float("nan")
    with StockfishEngine(stockfish_path) as engine:
        stockfish = engine.evaluate_fen(fen, depth=depth)
    print(f"FEN:       {fen}")
    print(f"Teacher:   {teacher:.3f}")
    if math.isnan(network):
        print("Network:   <no neural_weights.json>")
    else:
        print(f"Network:   {network:.3f}")
    print(f"Stockfish: {stockfish:.3f}")


def distill(
    stockfish_path: str,
    depth: int = DEFAULT_STOCKFISH_DEPTH,
    epochs: int = DEFAULT_DISTILL_EPOCHS,
    blend: float = DEFAULT_STOCKFISH_BLEND,
    fen_source: str | None = None,
    limit: int | None = None,
    sample_size: int | None = None,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = DEFAULT_RANDOM_SEED,
    use_curated: bool = True,
) -> None:
    extra_fens = import_fens(fen_source, sample_size=sample_size, seed=seed) if fen_source else None
    if extra_fens:
        print(f"imported_fens={len(extra_fens)} from {fen_source}")
    dataset, samples = build_stockfish_supervised_dataset(
        stockfish_path=stockfish_path,
        depth=depth,
        blend=blend,
        extra_fens=extra_fens,
        limit=limit,
        seed=seed,
        use_curated=use_curated,
    )
    print(f"distill positions: {len(samples)}")
    if not dataset:
        raise RuntimeError("No valid Stockfish-supervised positions were collected.")
    preview = samples[: min(5, len(samples))]
    for sample in preview:
        print(
            f"sample teacher={sample.teacher:.3f} stockfish={sample.stockfish:.3f} "
            f"blended={sample.blended:.3f} fen={sample.fen}"
        )
    train(
        dataset=dataset,
        epochs=epochs,
        base_lr=0.005,
        validation_split=validation_split,
        seed=seed,
    )


def train_lichess(
    fen_source: str,
    sample_size: int | None = None,
    epochs: int = DEFAULT_DISTILL_EPOCHS,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    labeled = import_labeled_lichess_positions(fen_source, sample_size=sample_size, seed=seed)
    print(f"imported_lichess_positions={len(labeled)} from {fen_source}")
    if not labeled:
        raise RuntimeError("No labeled Lichess positions were found in the source.")
    dataset = build_dataset_from_labeled_positions(labeled)
    for fen, score in labeled[: min(5, len(labeled))]:
        print(f"sample lichess_score={score:.3f} fen={fen}")
    train(
        dataset=dataset,
        epochs=epochs,
        base_lr=0.005,
        validation_split=validation_split,
        seed=seed,
    )


def load_weights(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def round_nested(value, digits: int = 6):
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_nested(item, digits) for key, item in value.items()}
    return value


def write_html_embed(weights_path: Path, html_path: Path) -> None:
    weights = round_nested(load_weights(weights_path), 6)
    payload = json.dumps(weights, separators=(",", ":"))
    html = html_path.read_text(encoding="utf-8")
    start_marker = '<script id="nn-weights" type="application/json">'
    end_marker = "</script>"
    start = html.find(start_marker)
    if start == -1:
        raise ValueError("Could not find nn-weights script tag in index.html")
    content_start = start + len(start_marker)
    end = html.find(end_marker, content_start)
    if end == -1:
        raise ValueError("Could not find end of nn-weights script tag in index.html")
    updated = html[:content_start] + payload + html[end:]
    html_path.write_text(updated, encoding="utf-8")


def embed() -> None:
    base = Path(__file__).resolve().parent
    weights_path = base / "neural_weights.json"
    html_path = base / "index.html"
    if not weights_path.exists():
        raise FileNotFoundError("neural_weights.json not found. Run: python train_chess_nn.py train")
    write_html_embed(weights_path, html_path)
    print(f"embedded weights from {weights_path.name} into {html_path.name}")


def run_network(x: list[float], weights: dict) -> float:
    normalized = []
    for i, value in enumerate(x):
        normalized.append((value - weights["feature_mean"][i]) / weights["feature_std"][i])

    h1 = []
    for i, row in enumerate(weights["w1"]):
        total = weights["b1"][i]
        for j, weight in enumerate(row):
            total += weight * normalized[j]
        h1.append(math.tanh(total))

    h2 = []
    for i, row in enumerate(weights["w2"]):
        total = weights["b2"][i]
        for j, weight in enumerate(row):
            total += weight * h1[j]
        h2.append(math.tanh(total))

    out = weights["b3"]
    for i, weight in enumerate(weights["w3"]):
        out += weight * h2[i]
    return out


def evaluate_loaded(fen: str, weights_path: Path) -> float:
    features, _ = evaluate_teacher(fen)
    weights = load_weights(weights_path)
    return run_network(features, weights)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"train", "eval", "embed", "refresh", "train-lichess", "compare", "distill", "distill-refresh"}:
        print(__doc__.strip())
        return 1

    weights_path = Path(__file__).with_name("neural_weights.json")
    if argv[1] == "train":
        train()
        return 0
    if argv[1] == "embed":
        embed()
        return 0
    if argv[1] == "refresh":
        train()
        embed()
        return 0
    if argv[1] == "train-lichess":
        fen_source = None
        sample_size = None
        epochs = DEFAULT_DISTILL_EPOCHS
        validation_split = DEFAULT_VALIDATION_SPLIT
        seed = DEFAULT_RANDOM_SEED
        i = 2
        while i < len(argv):
            if argv[i] in {"--fen-file", "--fen-dir"} and i + 1 < len(argv):
                fen_source = argv[i + 1]
                i += 2
            elif argv[i] == "--sample-size" and i + 1 < len(argv):
                sample_size = int(argv[i + 1])
                i += 2
            elif argv[i] == "--epochs" and i + 1 < len(argv):
                epochs = int(argv[i + 1])
                i += 2
            elif argv[i] == "--validation-split" and i + 1 < len(argv):
                validation_split = float(argv[i + 1])
                i += 2
            elif argv[i] == "--seed" and i + 1 < len(argv):
                seed = int(argv[i + 1])
                i += 2
            else:
                print(f"Unknown argument: {argv[i]}")
                return 1
        if not fen_source:
            print("Provide --fen-file or --fen-dir pointing to a Lichess eval source.")
            return 1
        train_lichess(
            fen_source=fen_source,
            sample_size=sample_size,
            epochs=epochs,
            validation_split=validation_split,
            seed=seed,
        )
        return 0
    if argv[1] == "compare":
        if len(argv) < 3:
            print('Provide a FEN to compare.')
            return 1
        stockfish_path = None
        depth = DEFAULT_STOCKFISH_DEPTH
        i = 3
        while i < len(argv):
            if argv[i] == "--stockfish" and i + 1 < len(argv):
                stockfish_path = argv[i + 1]
                i += 2
            elif argv[i] == "--depth" and i + 1 < len(argv):
                depth = int(argv[i + 1])
                i += 2
            else:
                print(f"Unknown argument: {argv[i]}")
                return 1
        compare_with_stockfish(argv[2], resolve_stockfish_path(stockfish_path), depth=depth)
        return 0
    if argv[1] in {"distill", "distill-refresh"}:
        stockfish_path = None
        depth = DEFAULT_STOCKFISH_DEPTH
        epochs = DEFAULT_DISTILL_EPOCHS
        blend = DEFAULT_STOCKFISH_BLEND
        fen_source = None
        limit = None
        sample_size = None
        validation_split = DEFAULT_VALIDATION_SPLIT
        seed = DEFAULT_RANDOM_SEED
        use_curated = True
        i = 2
        while i < len(argv):
            if argv[i] == "--stockfish" and i + 1 < len(argv):
                stockfish_path = argv[i + 1]
                i += 2
            elif argv[i] == "--depth" and i + 1 < len(argv):
                depth = int(argv[i + 1])
                i += 2
            elif argv[i] == "--epochs" and i + 1 < len(argv):
                epochs = int(argv[i + 1])
                i += 2
            elif argv[i] == "--blend" and i + 1 < len(argv):
                blend = float(argv[i + 1])
                i += 2
            elif argv[i] == "--fen-file" and i + 1 < len(argv):
                fen_source = argv[i + 1]
                i += 2
            elif argv[i] == "--fen-dir" and i + 1 < len(argv):
                fen_source = argv[i + 1]
                i += 2
            elif argv[i] == "--limit" and i + 1 < len(argv):
                limit = int(argv[i + 1])
                i += 2
            elif argv[i] == "--sample-size" and i + 1 < len(argv):
                sample_size = int(argv[i + 1])
                i += 2
            elif argv[i] == "--validation-split" and i + 1 < len(argv):
                validation_split = float(argv[i + 1])
                i += 2
            elif argv[i] == "--seed" and i + 1 < len(argv):
                seed = int(argv[i + 1])
                i += 2
            elif argv[i] == "--import-only":
                use_curated = False
                i += 1
            else:
                print(f"Unknown argument: {argv[i]}")
                return 1
        distill(
            stockfish_path=resolve_stockfish_path(stockfish_path),
            depth=depth,
            epochs=epochs,
            blend=blend,
            fen_source=fen_source,
            limit=limit,
            sample_size=sample_size,
            validation_split=validation_split,
            seed=seed,
            use_curated=use_curated,
        )
        if argv[1] == "distill-refresh":
            embed()
        return 0

    if len(argv) < 3:
        print("Provide a FEN to evaluate.")
        return 1
    if not weights_path.exists():
        print("neural_weights.json not found. Run: python train_chess_nn.py train")
        return 1
    score = evaluate_loaded(argv[2], weights_path)
    print(f"{score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
