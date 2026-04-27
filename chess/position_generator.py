import chess
import chess.engine
import random
import json
import zstandard as zstd
from tqdm import tqdm
from pathlib import Path

# =========================
# CONFIG
# =========================
STOCKFISH_PATH = r"C:\Users\1\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe" 
NUM_POSITIONS = 100000
OUTPUT_FILE = "lichess_db_eval.jsonl.zst"

# =========================
# GENERATION
# =========================

def generate_position():
    # Keep it simple to ensure it never hangs
    board = chess.Board()
    # Randomly play 10 to 60 moves
    moves = random.randint(10, 60)
    for _ in range(moves):
        if board.is_game_over(): break
        board.push(random.choice(list(board.legal_moves)))
    
    # Randomly remove some pieces occasionally to simulate endgames
    if random.random() < 0.2:
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING and random.random() < 0.5:
                board.remove_piece_at(square)
    
    if not board.is_valid(): return chess.Board()
    return board

# =========================
# MAIN
# =========================

def generate_dataset():
    print(f"Starting Engine...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 4, "Hash": 128})
    
    print(f"Generating {NUM_POSITIONS} positions...")
    
    with open(OUTPUT_FILE, "wb") as f:
        compressor = zstd.ZstdCompressor(level=3)
        with compressor.stream_writer(f) as writer:
            
            for _ in tqdm(range(NUM_POSITIONS), desc="Progress"):
                board = generate_position()
                
                # Use a slightly longer time limit to ensure we get a result
                try:
                    # engine.analyse returns a list when MultiPV is 1 or a dict depending on call
                    # .analyse is safer than .evaluate
                    info = engine.analyse(board, chess.engine.Limit(time=0.03))
                    
                    # Robust extraction of score
                    # info is usually a list of dicts from .analyse
                    if isinstance(info, list):
                        res = info[0]
                    else:
                        res = info
                        
                    score = res["score"].relative
                    
                    if score.is_mate():
                        val = 10000 if score.mate() > 0 else -10000
                    else:
                        val = score.score()
                        
                    if val is None: continue

                    record = {
                        "fen": board.fen(),
                        "evals": [{"pvs": [{"cp": val}]}]
                    }
                    
                    writer.write((json.dumps(record) + "\n").encode("utf-8"))
                except Exception:
                    continue

    engine.quit()
    print("Done!")

if __name__ == "__main__":
    generate_dataset()