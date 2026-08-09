import sys
from dataset_dl import clean_text

with open(r'E:\training\data2\extracted\2663.html', 'r', encoding='utf-8') as f:
    html = f.read()

text = clean_text(html)
print(f'Text length: {len(text)}', file=sys.stderr)
print('First 500 chars:', file=sys.stderr)
print(text[:500], file=sys.stderr)
print('Last 200 chars:', file=sys.stderr)
print(text[-200:], file=sys.stderr)
