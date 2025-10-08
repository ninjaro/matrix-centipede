#!/usr/bin/env python3
import sys
import re
from collections import defaultdict

import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r'^(?P<name>\S+).*?\bFLOPs=(?P<gflops>[\deE+\-\.]+)G/s\b'
)

def parse_alg(name: str) -> str:
    parts = name.split('/')
    if parts[0] == 'BM_DM' and len(parts) > 1:
        return parts[1]
    if parts[0].startswith('BM_Eigen'):
        return 'eigen'
    return parts[0]

def parse_log(path: str):
    series = defaultdict(lambda: defaultdict(list))
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            m = LINE_RE.match(raw.strip())
            if not m:
                continue
            name = m.group('name')
            if name.endswith(('_stddev', '_cv', '_median')):
                continue
            if name.endswith('_mean'):
                name = name[:-5]
            n_match = re.search(r'/n:(\d+)', name)
            if not n_match:
                continue
            n = int(n_match.group(1))
            gflops = float(m.group('gflops'))
            alg = parse_alg(name)
            series[alg][n].append(gflops)

    collapsed = {}
    for alg, by_n in series.items():
        xs = sorted(by_n.keys())
        ys = [sum(by_n[k]) / len(by_n[k]) for k in xs]
        collapsed[alg] = (xs, ys)
    return collapsed

def main(argv):
    if len(argv) < 3:
        return 0
    in_path = argv[1]
    out_path = argv[2]

    data = parse_log(in_path)

    plt.figure()
    for alg, (xs, ys) in sorted(data.items()):
        plt.plot(xs, ys, marker='o', label=alg)
    plt.xlabel('n')
    plt.ylabel('GFLOPs/s')
    plt.title('GEMM Benchmark (GFLOPs/s vs n)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)

    return 0

if __name__ == '__main__':
    main(sys.argv)
