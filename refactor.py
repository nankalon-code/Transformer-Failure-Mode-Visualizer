import re
import os

with open('src/App.tsx', 'r', encoding='utf-8') as f:
    app_code = f.read()

# Extract panels
panels = {}
for i in range(1, 6):
    pattern = r"function Panel" + str(i) + r"\(\) \{.*?(?=\nfunction Panel|\nexport default function App)"
    match = re.search(pattern, app_code, re.DOTALL)
    if match:
        panels[i] = match.group(0)

os.makedirs('src/components', exist_ok=True)

imports_needed = """import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { norm, randomMatrix, normalizeRows, matMul, transpose, softmaxRows, entropyRows, getStats } from '../utils/math';
"""

for i, code in panels.items():
    # Make them exported properly
    code = code.replace(f"function Panel{i}()", f"export default function Panel{i}()")
    with open(f'src/components/Panel{i}.tsx', 'w', encoding='utf-8') as f:
        f.write(imports_needed + "\n" + code)

# Create new App.tsx
new_app = """import { useState } from 'react';
import Panel1 from './components/Panel1';
import Panel2 from './components/Panel2';
import Panel3 from './components/Panel3';
import Panel4 from './components/Panel4';
import Panel5 from './components/Panel5';

"""

app_match = re.search(r"export default function App\(\) \{.*", app_code, re.DOTALL)
if app_match:
    # Need to update the comp tags if they were using <Panel1 /> etc, which they are.
    new_app += app_match.group(0)

with open('src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(new_app)

print("Refactoring complete.")
