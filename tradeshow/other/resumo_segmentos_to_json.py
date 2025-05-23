import os
import json

input_path = os.path.join(os.path.dirname(__file__), 'resumo_segmentos.txt')
output_path = os.path.join(os.path.dirname(__file__), 'resumo_segmentos.json')

segments = []
with open(input_path, encoding='utf-8') as f:
    lines = [line.rstrip('\n') for line in f]

current_nome = None
current_resumo = []
for line in lines + ['']:
    if not line.strip():
        # Blank line or end of file: flush if we have a segment
        if current_nome and current_resumo:
            segments.append({
                'nome': current_nome,
                'resumo': ' '.join(current_resumo).strip()
            })
            current_nome = None
            current_resumo = []
        continue
    if current_nome is None:
        current_nome = line.strip()
        current_resumo = []
    else:
        current_resumo.append(line.strip())

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(segments)} segments to {output_path}") 