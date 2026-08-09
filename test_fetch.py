import sys, subprocess, time, hashlib, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def fetch(url, out_file):
    cmd = ["curl", "-s", "-L", "-A", UA,
           "-H", "Accept: text/html,application/xhtml+xml",
           "-H", "Accept-Language: bg-BG,bg;q=0.9",
           "--connect-timeout", "10", "--max-time", "60",
           url, "-o", out_file]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and Path(out_file).stat().st_size > 1000

def clean_text(html_text):
    start = html_text.find('Към текста')
    if start == -1:
        start = html_text.find('id="text-content"')
    end = html_text.rfind('Към началото')
    if end == -1:
        end = html_text.find('</body>')
    if start == -1 or end <= start:
        return None

    raw = html_text[start:end]
    raw = re.sub(r'<br\s*/?>', '\n', raw)
    raw = re.sub(r'<p[^>]*>', '\n', raw)
    raw = re.sub(r'</p>', '', raw)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = re.sub(r'\n\s*\n\s*\n+', '\n\n', raw)

    lines = []
    found = False
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^(Сваляне|Включено|Традиционна|Характеристика|Вашата)', line):
            continue
        if any(kw in line for kw in ['Към навигацията', 'Към текста', 'Моята библиотека',
                                      'Читалня', 'Работилница', 'Колекции', 'Проекти',
                                      'Читателите', 'са прочели и']):
            continue
        if re.match(r'^[1-6]\s*[—–-]', line) and len(line) < 40:
            continue
        if not found and len(line) < 15:
            continue
        if not found and len(line) >= 15:
            found = True
        if found and 15 <= len(line) <= 2048:
            lines.append(line)
        if line in ('Край', 'Конец', 'КРАЙ'):
            break
    return '\n'.join(lines)

# Test with a few text IDs
ids = ['10997', '13220', '14020', '14111', '14890']
out_dir = Path('E:\\training\\data2\\extracted_test')
out_dir.mkdir(parents=True, exist_ok=True)

success = 0
texts = []
seen = set()

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    for idx, text_id in enumerate(ids):
        out_file = out_dir / f"{text_id}.html"
        url = f"http://chitanka.info/text/{text_id}"
        future = executor.submit(fetch, url, str(out_file))
        futures.append((text_id, future))
        time.sleep(1)  # 1 second delay between requests

    for text_id, future in futures:
        out_file = out_dir / f"{text_id}.html"
        if future.result():
            success += 1
            html = out_file.read_text(encoding='utf-8')
            text = clean_text(html)
            if text and len(text) > 100:
                h = hashlib.md5(text.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    texts.append(text)
                    print(f"OK {text_id}: {len(text)} chars")
            else:
                print(f"CLEAR {text_id}: clean_text failed")
        else:
            print(f"FAIL {text_id}")

print(f"\nSuccess: {success}/{len(ids)}")
print(f"Unique texts: {len(texts)}")
if texts:
    out_file = out_dir / "output.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n\n")
    print(f"Written to {out_file}: {out_file.stat().st_size} bytes")
