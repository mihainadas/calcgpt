# Arithmetic Expression Generator

A modern CLI tool for generating arithmetic expressions (addition and subtraction) with configurable parameters and filtering options.

## Features

- **Configurable value ranges**: Set minimum and maximum values for operands
- **Digit filtering**: Restrict expressions to use only specific digits (e.g., only 1, 2, 3)
- **Range support**: Use ranges like "1-5" for allowed digits
- **Operation filtering**: Generate only addition or only subtraction expressions
- **Output limiting**: Set maximum number of expressions to generate
- **Verbose output**: See detailed generation parameters
- **Modern CLI**: Full argument parsing with help and examples

## Usage

### Basic Usage

```bash
# Generate expressions with default settings (0-100)
python3 datagen.py -o data/basic.txt

# Generate with custom maximum value
python3 datagen.py -o data/small.txt -m 10
```

### Digit Filtering

```bash
# Only use digits 1, 2, 3 in expressions
python3 datagen.py -o data/filtered.txt -m 50 -d "1,2,3"

# Use digit ranges
python3 datagen.py -o data/range.txt -m 100 -d "1-5"

# Mix individual digits and ranges
python3 datagen.py -o data/mixed.txt -m 50 -d "1-3,7,9"
```

### Operation Filtering

```bash
# Generate only addition expressions
python3 datagen.py -o data/add_only.txt -m 20 --no-subtraction

# Generate only subtraction expressions
python3 datagen.py -o data/sub_only.txt -m 20 --no-addition
```

### Advanced Options

```bash
# Set minimum value, limit output, and use verbose mode
python3 datagen.py -o data/advanced.txt -m 100 --min-value 10 --max-expressions 500 -v
```

## Command Line Options

```
-o, --output OUTPUT              Output file path (required)
-m, --max-value MAX_VALUE        Maximum value for operands (default: 100)
--min-value MIN_VALUE            Minimum value for operands (default: 0)
-d, --allowed-digits DIGITS      Comma-separated allowed digits or ranges
--max-expressions N              Maximum number of expressions to generate
--no-addition                    Exclude addition expressions
--no-subtraction                 Exclude subtraction expressions
-v, --verbose                    Enable verbose output
-h, --help                       Show help message
```

## How Digit Filtering Works

When you specify allowed digits (e.g., `-d "1,2,3"`), the generator:

1. Only uses operands that contain exclusively those digits (1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33, etc.)
2. Only includes expressions where the result also contains only those digits
3. Supports both individual digits (`1,2,3`) and ranges (`1-3`)

## Examples

```bash
# Generate simple expressions for small numbers
./datagen.py -o data/simple.txt -m 5 -v

# Generate expressions using only digits 1-3
./datagen.py -o data/restricted.txt -m 50 -d "1-3" -v

# Generate 100 addition-only expressions with minimum value 5
./datagen.py -o data/addition.txt -m 20 --min-value 5 --max-expressions 100 --no-subtraction -v
```

## Requirements

- Python 3.7+ (for type hints and pathlib support)
- No external dependencies (uses only standard library)

## Output Format

Each line in the output file contains one arithmetic expression in the format:
```
operand1+operand2=result
operand1-operand2=result
```

For example:
```
1+2=3
5-3=2
12+21=33
``` 