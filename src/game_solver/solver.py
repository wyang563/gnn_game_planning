import numpy as np
from julia.api import Julia

# CRITICAL: code in this file must be run with python-jl as opposed to python due to julia compatibility issues

class NashSolver:
    def __init__(self):
        self.setup_julia()
    
    def setup_julia(self):
        try:
            from julia import Main
            from julia import Pkg
            
            # Always setup project directory in Julia (needed for each Python process)
            project_dir = "/home/alex/gnn_game_planning"
            Main.eval(f'ENV["JULIA_PROJECT"] = "{project_dir}"')
            Pkg.activate(project_dir)
            
            Pkg.add("MixedComplementarityProblems")
            print("Julia environment setup successful!")
            self.julia_initialized = True

            Main.eval("using MixedComplementarityProblems")
            self.julia = Main
            print("MixedComplementarityProblems.jl loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to setup Julia environment: {e}")

if __name__ == "__main__":
    solver = NashSolver()