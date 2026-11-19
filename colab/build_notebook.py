"""Convert qfc_colab.py (# %% cells) into quantum_fit_check_colab.ipynb. Usage: python colab/build_notebook.py"""
import json, re, pathlib
here = pathlib.Path(__file__).parent; src = (here / 'qfc_colab.py').read_text(encoding='utf8'); cells = []
for block in re.split(r'^# %%', src, flags=re.M)[1:]:
    if block.startswith(' [markdown]'):
        body = block.split('\n', 1)[1]; cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip().splitlines())})
    else:
        body = block.split('\n', 1)[1].strip('\n').replace('# !pip -q install', '!pip -q install')
        body = body.replace("CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'data.csv'", "CSV_PATH = 'data.csv'").replace("TASK = sys.argv[2] if len(sys.argv) > 2 else 'cls'", "TASK = 'cls'")
        body = body.replace('df = pd.read_csv(CSV_PATH)\nprint(df.shape); df.head()', '')
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': body.strip('\n')})
cells.insert(4, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': "# 上傳 CSV（Colab 會跳出選檔視窗；本機執行則直接讀 CSV_PATH）\ntry:\n    from google.colab import files\n    up = files.upload()\n    if up: CSV_PATH = next(iter(up))\nexcept ImportError:\n    pass\nfor _enc in ('utf-8-sig', 'utf-8', 'big5', 'cp950', 'utf-16'):\n    try:\n        df = pd.read_csv(CSV_PATH, encoding=_enc); break\n    except UnicodeDecodeError:\n        pass\nelse:\n    df = pd.read_csv(CSV_PATH, encoding='utf-8', errors='replace')\nprint(df.shape); df.head()"})
nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}}, 'cells': cells}
(here / 'quantum_fit_check_colab.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf8'); print(len(cells), 'cells')
