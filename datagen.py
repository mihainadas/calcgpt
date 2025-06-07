#!/usr/bin/env python3
"""
Arithmetic Expression Generator

A CLI tool for generating arithmetic expressions (addition and subtraction)
with configurable parameters and filtering options.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set, Generator, Tuple


def contains_only_allowed_digits(number: int, allowed_digits: Set[str]) -> bool:
    """Check if a number contains only the allowed digits."""
    return all(digit in allowed_digits for digit in str(number))


def generate_valid_numbers(max_value: int, allowed_digits: Set[str] = None) -> List[int]:
    """Generate list of valid numbers based on constraints."""
    if allowed_digits is None:
        return list(range(max_value + 1))
    
    valid_numbers = []
    for i in range(max_value + 1):
        if contains_only_allowed_digits(i, allowed_digits):
            valid_numbers.append(i)
    
    return valid_numbers


def generate_expressions(
    max_value: int,
    allowed_digits: Set[str] = None,
    include_addition: bool = True,
    include_subtraction: bool = True,
    min_value: int = 0
) -> Generator[str, None, None]:
    """Generate arithmetic expressions based on given parameters."""
    
    valid_numbers = generate_valid_numbers(max_value, allowed_digits)
    
    # Filter by minimum value if specified
    if min_value > 0:
        valid_numbers = [n for n in valid_numbers if n >= min_value]
    
    if not valid_numbers:
        raise ValueError("No valid numbers found with the given constraints")
    
    for i in valid_numbers:
        for j in valid_numbers:
            # Addition expressions
            if include_addition:
                result = i + j
                # Check if result also satisfies digit constraints
                if allowed_digits is None or contains_only_allowed_digits(result, allowed_digits):
                    yield f"{i}+{j}={result}"
            
            # Subtraction expressions (only when i >= j to avoid negative results)
            if include_subtraction and i >= j:
                result = i - j
                # Check if result also satisfies digit constraints
                if allowed_digits is None or contains_only_allowed_digits(result, allowed_digits):
                    yield f"{i}-{j}={result}"


def write_expressions_to_file(
    expressions: Generator[str, None, None],
    output_path: Path,
    max_expressions: int = None
) -> int:
    """Write expressions to file and return count of expressions written."""
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_path, 'w') as f:
        for expression in expressions:
            f.write(expression + '\n')
            count += 1
            
            if max_expressions and count >= max_expressions:
                break
    
    return count


def parse_digit_set(digit_string: str) -> Set[str]:
    """Parse comma-separated digits into a set."""
    if not digit_string:
        return None
    
    digits = set()
    for part in digit_string.split(','):
        part = part.strip()
        if '-' in part and len(part) == 3:  # Range like "1-3"
            start, end = part.split('-')
            if start.isdigit() and end.isdigit():
                for i in range(int(start), int(end) + 1):
                    digits.add(str(i))
        elif part.isdigit():
            digits.add(part)
        else:
            raise ValueError(f"Invalid digit specification: {part}")
    
    return digits


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate arithmetic expressions with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -o data/simple.txt -m 10
  %(prog)s -o data/filtered.txt -m 50 -d "1,2,3"
  %(prog)s -o data/range.txt -m 100 -d "1-5" --min-value 1
  %(prog)s -o data/addition_only.txt -m 20 --no-subtraction
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output file path'
    )
    
    # Optional arguments
    parser.add_argument(
        '-m', '--max-value',
        type=int,
        default=100,
        help='Maximum value for operands (default: 100)'
    )
    
    parser.add_argument(
        '--min-value',
        type=int,
        default=0,
        help='Minimum value for operands (default: 0)'
    )
    
    parser.add_argument(
        '-d', '--allowed-digits',
        type=str,
        help='Comma-separated allowed digits or ranges (e.g., "1,2,3" or "1-3,7,9")'
    )
    
    parser.add_argument(
        '--max-expressions',
        type=int,
        help='Maximum number of expressions to generate'
    )
    
    parser.add_argument(
        '--no-addition',
        action='store_true',
        help='Exclude addition expressions'
    )
    
    parser.add_argument(
        '--no-subtraction',
        action='store_true',
        help='Exclude subtraction expressions'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.max_value < 0:
        print("Error: max-value must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    if args.min_value < 0:
        print("Error: min-value must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    if args.min_value > args.max_value:
        print("Error: min-value cannot be greater than max-value", file=sys.stderr)
        sys.exit(1)
    
    if args.no_addition and args.no_subtraction:
        print("Error: Cannot exclude both addition and subtraction", file=sys.stderr)
        sys.exit(1)
    
    # Parse allowed digits
    try:
        allowed_digits = parse_digit_set(args.allowed_digits) if args.allowed_digits else None
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Generating expressions with parameters:")
        print(f"  Max value: {args.max_value}")
        print(f"  Min value: {args.min_value}")
        print(f"  Allowed digits: {sorted(allowed_digits) if allowed_digits else 'all'}")
        print(f"  Include addition: {not args.no_addition}")
        print(f"  Include subtraction: {not args.no_subtraction}")
        print(f"  Output file: {args.output}")
        if args.max_expressions:
            print(f"  Max expressions: {args.max_expressions}")
    
    try:
        # Generate expressions
        expressions = generate_expressions(
            max_value=args.max_value,
            allowed_digits=allowed_digits,
            include_addition=not args.no_addition,
            include_subtraction=not args.no_subtraction,
            min_value=args.min_value
        )
        
        # Write to file
        count = write_expressions_to_file(
            expressions,
            args.output,
            args.max_expressions
        )
        
        print(f"Generated {count} expressions and saved to {args.output}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()