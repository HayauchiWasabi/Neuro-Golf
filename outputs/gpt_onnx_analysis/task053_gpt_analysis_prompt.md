You are helping optimize a NeuroGolf ONNX model.

Rules:
- Input tensor: float32 [1, 10, 30, 30], name 'input'.
- Output tensor: float32 [1, 10, 30, 30], name 'output'.
- Static tensor shapes only.
- One input and one output only.
- Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.
- Optimize score by reducing memory + params.
- Assume public examples pass unless told otherwise.

Requested output:
1. Summarize what the current ONNX appears to do.
2. Identify redundant or expensive parts.
3. Propose safe rewrite candidates ranked by risk and likely score gain.
4. Do not write code yet.

Task summary:
- task: task053
- train examples: 4
- test examples: 2
- arc-gen examples: 54
- color counts: {0: 770, 1: 160, 2: 150}
- example shapes: [{'subset': 'train', 'index': 0, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'train', 'index': 1, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'train', 'index': 2, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'train', 'index': 3, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'test', 'index': 0, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'test', 'index': 1, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 0, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 1, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 2, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 3, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 4, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 5, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 6, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 7, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 8, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 9, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 10, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 11, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 12, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 13, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 14, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 15, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 16, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 17, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 18, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 19, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 20, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 21, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 22, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 23, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 24, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 25, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 26, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 27, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 28, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 29, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 30, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 31, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 32, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 33, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 34, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 35, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 36, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 37, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 38, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 39, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 40, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 41, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 42, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 43, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 44, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 45, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 46, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 47, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 48, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 49, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 50, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 51, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 52, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}, {'subset': 'arc-gen', 'index': 53, 'input_shape': '3x3', 'output_shape': '3x3', 'input_cells': 9, 'output_cells': 9}]

# ONNX analysis: task053.onnx

## Basic
- path: /Users/kaiikeda/Programming/Kaggle/Neuro Golf/solution/6410.88/task053.onnx
- filesize: 175 bytes
- ir_version: 8
- opset_imports: [{'domain': '', 'version': 10}]
- inputs: [{'name': 'input', 'elem_type': 'FLOAT', 'shape': [1, 10, 30, 30]}]
- outputs: [{'name': 'output', 'elem_type': 'FLOAT', 'shape': [1, 10, 30, 30]}]
- nodes: 1
- initializers: 1
- op_counts: {'Gather': 1}

## Score
- status: ok
- memory: 0
- params: 30
- cost: 30.0
- score: 21.598802618337846
- score_error: 

## Initializers
- v0: dtype=INT64, dims=[30], numel=30, bytes=240, sample=[2, 0, 1, 29, 29, 29, 29, 29, 29, 29, 29, 29]

## Nodes
- #0 Gather name= inputs=['input', 'v0'] outputs=['output'] attrs={'axis': 2}