import ast
import astunparse

def refactor_code(input_file, output_file):
    with open(input_file, 'r') as f:
        source = f.read()

    # Remove all top-level multi-line strings (Markdown cells from Colab)
    # and remove IPython magic lines like !pip install
    lines = source.split('\n')
    lines = [line if not line.strip().startswith('!') else f"# {line}" for line in lines]
    source = '\n'.join(lines)
    tree = ast.parse(source)

    imports = []
    from_imports = []
    classes = {}
    functions = {}
    assignments = []
    other_statements = []

    for node in tree.body:
        # Ignore standalone docstrings / multi-line strings
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Constant)):
            continue
            
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in [imp.names[0].name for imp in imports]:
                    imports.append(node)
        elif isinstance(node, ast.ImportFrom):
            # simplify: just keep them if they are unique
            from_imports.append(node)
        elif isinstance(node, ast.ClassDef):
            # Overwrite earlier definition with the latest one
            classes[node.name] = node
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, ast.Assign):
            assignments.append(node)
        else:
            other_statements.append(node)

    # Order of reconstructed file:
    # 1. Imports
    # 2. Geometry & Solvers
    # 3. Models (PeriodicUNet, FNO2d, GeoFNO, WNO)
    # 4. Datasets & Trainers
    # 5. Functions
    
    new_body = []
    new_body.extend(imports)
    new_body.extend(from_imports)

    # We can categorize classes based on names
    geometry_solvers = ['TorusGeometry', 'TorusWaveSolverRK4', 'TorusSpectralSolver', 'TorusAcousticSimulator']
    models = ['SpectralConv2d', 'FNO2d', 'DoubleConv', 'Down', 'Up', 'OutConv', 'PeriodicUNet', 
              'BaseFNO2d', 'DiffeomorphismNet', 'GeoFNO', 'WaveletConv2d', 'WaveletNeuralOperator']
    datasets_trainers = ['TorusWaveDataset', 'DataDrivenTrainer', 'ManifoldEvaluator', 
                         'ScaledPhysicsLoss', 'FNOEvaluator']

    # Add matched classes
    for cat in [geometry_solvers, models, datasets_trainers]:
        for name in cat:
            if name in classes:
                new_body.append(classes[name])
                
    # Add any remaining classes
    for name, node in classes.items():
        if name not in geometry_solvers and name not in models and name not in datasets_trainers:
            new_body.append(node)
            
    # Add functions
    for name, node in functions.items():
        new_body.append(node)

    # Reconstruct AST
    new_tree = ast.Module(body=new_body, type_ignores=[])
    
    # Generate source code
    cleaned_source = astunparse.unparse(new_tree)

    with open(output_file, 'w') as f:
        f.write("# Cleaned NOMAD CHORUS script\n")
        f.write(cleaned_source)

if __name__ == '__main__':
    refactor_code('nomad_chorus.py', 'nomad_chorus_clean.py')
    print("Refactoring complete.")
