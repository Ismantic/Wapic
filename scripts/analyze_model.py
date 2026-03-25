#!/usr/bin/env python3
"""Analyze model file to understand size breakdown."""
import collections

counts = collections.Counter()
sizes = collections.Counter()

with open('data/model.crf', 'r') as f:
    first = f.readline()  # #Patterns#11#1
    for _ in range(11):
        f.readline()

    # Label trie
    label_hdr = f.readline().strip()
    label_count = int(label_hdr.split('#Trie#')[1])
    print(f'Labels: {label_count}')
    for _ in range(label_count):
        line = f.readline().strip()
        print(f'  {line}')

    # Observation trie
    obs_hdr = f.readline().strip()
    obs_count = int(obs_hdr.split('#Trie#')[1])
    print(f'\nTotal observations: {obs_count:,}')

    for i in range(obs_count):
        line = f.readline()
        colon = line.index(':')
        value = line[colon+1:].rstrip(',\n')
        sep = value.find(':')
        if sep >= 0:
            pid = value[:sep]
        else:
            pid = value
        counts[pid] += 1
        sizes[pid] += len(line.encode())

print('\nObservations per pattern:')
print(f'  {"pattern":8s}  {"obs":>10s}  {"weights":>10s}  {"trie size":>10s}')
print(f'  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*10}')
total_weights = 0
for pid, cnt in sorted(counts.items()):
    sz = sizes[pid]
    if pid == 'b':
        w = cnt * 16  # Y*Y
    else:
        w = cnt * 4   # Y
    total_weights += w
    print(f'  {pid:8s}  {cnt:>10,}  {w:>10,}  {sz/1024/1024:>8.1f} MB')

total_obs = sum(counts.values())
total_trie = sum(sizes.values())
print(f'  {"TOTAL":8s}  {total_obs:>10,}  {total_weights:>10,}  {total_trie/1024/1024:>8.1f} MB')
print(f'\n  Active weights: 879,605 / {total_weights:,} = {879605/total_weights*100:.1f}%')
