import numpy as np
from julia.api import Julia

class NashSolver:
    def __init__(self, julia_initialized):
        self.julia_initialized = julia_initialized
        self.setup_julia()
    
    def setup_julia(self):
        if not self.julia_initialized:
            try:
                # Initialize Julia with compiled_modules=False to work around static linking issues
                jl = Julia(compiled_modules=False)
                
                from julia import Main
                from julia import Pkg
                
                # setup project directory in Julia
                project_dir = "/home/alex/gnn_game_planning"
                Main.eval(f'ENV["JULIA_PROJECT"] = "{project_dir}"')
                Pkg.activate(project_dir)

                print("Julia environment setup successful!")
                self.julia_initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Julia environment: {e}")
        
        try:
            from julia import Main

            Main.eval("using MixedComplementarityProblems")
            self.julia = Main
            print("MixedComplementarityProblems.jl loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load MixedComplementarityProblems: {e}")

if __name__ == "__main__":
    solver = NashSolver(julia_initialized=False)