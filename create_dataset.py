from num2words import num2words
import random

ONES = list(range(0, 10))
TENS = list(range(10, 100))
HUNDREDS = list(range(100, 1000))
ALL_NUMBERS = ONES + TENS + HUNDREDS

dataset = []

def format_example(a, b, op):
    if op == '+':
        result = a + b
        op_word = 'plus'
    else:
        result = a - b
        op_word = 'minus'

    expr = f"{a} {op} {b} = {result}"
    sentence = f"{num2words(a)} {op_word} {num2words(b)} is {num2words(result)}"
    return f"{expr}, {sentence}"

for i in range(1, 1000):
    choosenot = random.choice(['not', ''])
    op = random.choice(['+', '-'])
    a = i if op == '+' else -i
    if choosenot == 'not':
        dataset.append(f"{-a} is not {num2words(a)}")
    else:
        dataset.append(f"{a} is {num2words(a)}")

for a in ONES:
    for b in ONES:
        for op in ['+', '-']:
            line = format_example(a, b, op)
            if line:
                dataset.append(line)

for a in TENS:
    b = random.choice(ONES)
    for op in ['+', '-']:
        line = format_example(a, b, op)
        if line:
            dataset.append(line)

for a in TENS:
    b = random.choice(TENS)
    for op in ['+', '-']:
        line = format_example(a, b, op)
        if line:
            dataset.append(line)

for _ in range(100):
    a = random.choice(HUNDREDS)
    b = random.choice(ALL_NUMBERS)
    op = random.choice(['+', '-'])
    line = format_example(a, b, op)
    if line:
        dataset.append(line)


# 저장
with open("basic_math_dataset.txt", "w", encoding="utf-8") as f:
    for line in dataset:
        f.write(line + "\n")

print("✅ Dataset 생성 완료: basic_math_dataset.txt")
