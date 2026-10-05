"""Lossless JSON and spreadsheet-safe CSV exports."""
import csv
import io
import json


def as_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'


def as_csv(result: dict) -> str:
    stream = io.StringIO(newline='')
    keys = ['rank', 'reaction_id', 'description', 'logit', 'sigmoid_score', 'selected']
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for option in sorted(result['options'], key=lambda x: x['rank']):
        row = {k: option[k] for k in keys}
        for key, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                row[key] = "'" + value
        writer.writerow(row)
    return stream.getvalue()
