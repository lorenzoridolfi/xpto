import os
import json

# Paths
input_segments_path = os.path.join('/Users/lorenzo/Sync/Source/AI/autogen/tradeshow', 'input', 'segments.json')
resumo_path = os.path.join('/Users/lorenzo/Sync/Source/AI/autogen/tradeshow/other', 'resumo_segmentos.json')
output_segments_path = os.path.join('/Users/lorenzo/Sync/Source/AI/autogen/tradeshow', 'other', 'segments.json')
# Use only the correct synthetic user schema in the schema folder:
schema_path = os.path.join('/Users/lorenzo/Sync/Source/AI/autogen/tradeshow', 'schema', 'segments_schema.json')
# If you need the synthetic user schema, use:
# synthetic_user_schema_path = os.path.join('/Users/lorenzo/Sync/Source/AI/autogen/tradeshow', 'schema', 'synthetic_user_schema.json')

# Load segments
with open(input_segments_path, encoding='utf-8') as f:
    segments_data = json.load(f)

# Load resumo
with open(resumo_path, encoding='utf-8') as f:
    resumo_list = json.load(f)
resumo_map = {r['nome']: r['resumo'] for r in resumo_list}

# Merge: replace 'descricao' with 'resumo' for each segment
for seg in segments_data['segmentos']:
    apelido = seg.get('apelido')
    if apelido in resumo_map:
        seg['descricao'] = resumo_map[apelido]

# Write merged segments
with open(output_segments_path, 'w', encoding='utf-8') as f:
    json.dump(segments_data, f, ensure_ascii=False, indent=2)
print(f"Merged segments written to {output_segments_path}")

# Update schema: remove 'resumo' field if present
with open(schema_path, encoding='utf-8') as f:
    schema = json.load(f)
segment_props = schema['properties']['segmentos']['items']['properties']
if 'resumo' in segment_props:
    del segment_props['resumo']
# Also remove from required if present
required = schema['properties']['segmentos']['items'].get('required', [])
if 'resumo' in required:
    required.remove('resumo')
with open(schema_path, 'w', encoding='utf-8') as f:
    json.dump(schema, f, ensure_ascii=False, indent=2)
print(f"Schema updated: 'resumo' field removed if present at {schema_path}") 