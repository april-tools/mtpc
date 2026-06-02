def logits(s, n, v, r):
    return s * n * v * r * 4 / (1024 ** 3)


def format_size(gb):
    if gb >= 1024:
        return f"{gb / 1024:.1f} TB"
    elif gb >= 1:
        return f"{gb:.1f} GB"
    else:
        return f"{gb * 1024:.1f} MB"


S = 8192
print("| V | N | R | Size (fp32) |")
print("|---|---|---|-------------|")
for v in [320, 32_000, 128_000]:
    for n in [8, 16]:
        for r in [1, 4, 8, 16, 32]:
            size = format_size(logits(S, n, v, r))
            print(f"| {v:,} | {n} | {r} | {size} |")
