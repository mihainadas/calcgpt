#!/usr/bin/env python3
"""
CalcGPT Dataset Generator - Arithmetic expression dataset creation tool

A powerful command-line interface for generating arithmetic expression datasets
with configurable parameters, filtering options, and analysis capabilities.

Author: Mihai NADAS
Version: 1.0.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Set, Generator, Tuple, Dict, Any

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Print the dataset generator banner"""
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                    CalcGPT DataGen                            ║
║                 Dataset Generation Tool                      ║
║                         v1.0.0                               ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

def contains_only_allowed_digits(number: int, allowed_digits: Set[str]) -> bool:
    """Check if a number contains only the allowed digits."""
    return all(digit in allowed_digits for digit in str(number))

def generate_valid_numbers(max_value: int, allowed_digits: Set[str] = None, verbose: bool = False) -> List[int]:
    """Generate list of valid numbers based on constraints."""
    if verbose:
        print(f"{Colors.CYAN}🔢 Generating valid numbers up to {max_value}...{Colors.ENDC}")
    
    if allowed_digits is None:
        numbers = list(range(max_value + 1))
        if verbose:
            print(f"{Colors.GREEN}✅ Generated {len(numbers)} numbers (all digits allowed){Colors.ENDC}")
        return numbers
    
    valid_numbers = []
    total_checked = 0
    
    for i in range(max_value + 1):
        total_checked += 1
        if contains_only_allowed_digits(i, allowed_digits):
            valid_numbers.append(i)
        
        # Progress indicator for large ranges
        if verbose and total_checked % 10000 == 0:
            print(f"{Colors.CYAN}   Checked {total_checked} numbers, found {len(valid_numbers)} valid...{Colors.ENDC}")
    
    if verbose:
        allowed_str = ', '.join(sorted(allowed_digits))
        print(f"{Colors.GREEN}✅ Generated {len(valid_numbers)} numbers using digits: {allowed_str}{Colors.ENDC}")
    
    return valid_numbers

def generate_expressions(
    max_value: int,
    allowed_digits: Set[str] = None,
    include_addition: bool = True,
    include_subtraction: bool = True,
    min_value: int = 0,
    verbose: bool = False
) -> Generator[str, None, None]:
    """Generate arithmetic expressions based on given parameters."""
    
    if verbose:
        print(f"{Colors.CYAN}🧮 Generating arithmetic expressions...{Colors.ENDC}")
    
    valid_numbers = generate_valid_numbers(max_value, allowed_digits, verbose)
    
    # Filter by minimum value if specified
    if min_value > 0:
        original_count = len(valid_numbers)
        valid_numbers = [n for n in valid_numbers if n >= min_value]
        if verbose:
            filtered_count = original_count - len(valid_numbers)
            print(f"{Colors.WARNING}⚠️ Filtered out {filtered_count} numbers below minimum value {min_value}{Colors.ENDC}")
    
    if not valid_numbers:
        raise ValueError(f"{Colors.FAIL}No valid numbers found with the given constraints{Colors.ENDC}")
    
    if verbose:
        operations = []
        if include_addition:
            operations.append("addition")
        if include_subtraction:
            operations.append("subtraction")
        print(f"{Colors.CYAN}🔧 Operations to include: {', '.join(operations)}{Colors.ENDC}")
        
        # Estimate total possible expressions
        total_combinations = len(valid_numbers) ** 2
        if include_addition and include_subtraction:
            estimated = total_combinations * 1.5  # Rough estimate accounting for subtraction constraints
        else:
            estimated = total_combinations
        print(f"{Colors.CYAN}📊 Estimated expressions to generate: ~{int(estimated):,}{Colors.ENDC}")
    
    generated_count = 0
    
    for i in valid_numbers:
        for j in valid_numbers:
            # Addition expressions
            if include_addition:
                result = i + j
                # Check if result also satisfies digit constraints
                if allowed_digits is None or contains_only_allowed_digits(result, allowed_digits):
                    generated_count += 1
                    yield f"{i}+{j}={result}"
            
            # Subtraction expressions (only when i >= j to avoid negative results)
            if include_subtraction and i >= j:
                result = i - j
                # Check if result also satisfies digit constraints
                if allowed_digits is None or contains_only_allowed_digits(result, allowed_digits):
                    generated_count += 1
                    yield f"{i}-{j}={result}"
            
            # Progress indicator for verbose mode
            if verbose and generated_count % 1000 == 0 and generated_count > 0:
                print(f"{Colors.CYAN}   Generated {generated_count:,} expressions...{Colors.ENDC}")

def write_expressions_to_file(
    expressions: Generator[str, None, None],
    output_path: Path,
    max_expressions: int = None,
    verbose: bool = False
) -> int:
    """Write expressions to file and return count of expressions written."""
    
    if verbose:
        print(f"{Colors.CYAN}📝 Writing expressions to: {output_path}{Colors.ENDC}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    start_time = time.time()
    
    try:
        with open(output_path, 'w') as f:
            for expression in expressions:
                f.write(expression + '\n')
                count += 1
                
                # Progress indicator
                if verbose and count % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    print(f"{Colors.CYAN}   Written {count:,} expressions ({rate:.0f}/sec)...{Colors.ENDC}")
                
                if max_expressions and count >= max_expressions:
                    if verbose:
                        print(f"{Colors.WARNING}⚠️ Reached maximum expression limit: {max_expressions:,}{Colors.ENDC}")
                    break
        
        elapsed = time.time() - start_time
        if verbose:
            rate = count / elapsed if elapsed > 0 else 0
            print(f"{Colors.GREEN}✅ Successfully wrote {count:,} expressions in {elapsed:.1f}s ({rate:.0f}/sec){Colors.ENDC}")
    
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error writing to file: {e}{Colors.ENDC}")
        raise
    
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

def generate_output_filename(
    max_value: int,
    min_value: int = 0,
    allowed_digits: Set[str] = None,
    include_addition: bool = True,
    include_subtraction: bool = True,
    max_expressions: int = None,
    output_dir: str = "datasets"
) -> str:
    """Generate descriptive filename based on parameters."""
    
    parts = ["ds-calcgpt"]
    
    # Add min and max values
    parts.append(f"min{min_value}")
    parts.append(f"max{max_value}")
    
    # Add digit constraints
    if allowed_digits:
        # Sort digits to ensure consistent naming
        sorted_digits = sorted([int(d) for d in allowed_digits])
        
        # Check if it's a continuous range
        if len(sorted_digits) > 1 and sorted_digits == list(range(sorted_digits[0], sorted_digits[-1] + 1)):
            parts.append(f"ds{sorted_digits[0]}_de{sorted_digits[-1]}")
        else:
            # Individual digits
            digits_str = "_".join(str(d) for d in sorted_digits)
            parts.append(f"digits{digits_str}")
    else:
        parts.append("alldigits")
    
    # Add operation type
    if include_addition and include_subtraction:
        parts.append("allops")
    elif include_addition:
        parts.append("add")
    elif include_subtraction:
        parts.append("sub")
    
    # Add max expressions if specified
    if max_expressions:
        parts.append(f"limit{max_expressions}")
    
    filename = "_".join(parts) + ".txt"
    return str(Path(output_dir) / filename)

def parse_filename_parameters(filename: str) -> Dict[str, Any]:
    """Parse auto-generated filename to extract the parameters used to generate it."""
    
    # Remove path and extension
    base_name = Path(filename).stem
    
    # Check if it matches our naming convention
    if not base_name.startswith('ds-calcgpt'):
        raise ValueError(f"Filename '{filename}' doesn't match expected naming convention (ds-calcgpt_...)")
    
    # Split by underscores and remove the prefix
    parts = base_name.split('_')[1:]  # Skip 'ds-calcgpt'
    
    if len(parts) < 4:  # Minimum: min, max, digits, operations
        raise ValueError(f"Filename '{filename}' doesn't have enough components")
    
    params = {
        'min_value': 0,
        'max_value': None,
        'allowed_digits': None,
        'include_addition': True,
        'include_subtraction': True,
        'max_expressions': None
    }
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Parse min value
        if part.startswith('min'):
            params['min_value'] = int(part[3:])
        
        # Parse max value
        elif part.startswith('max'):
            params['max_value'] = int(part[3:])
        
        # Parse digit constraints
        elif part == 'alldigits':
            params['allowed_digits'] = None
        
        elif part.startswith('ds') and i + 2 < len(parts) and parts[i + 1].startswith('de'):
            # Range format: ds1_de5
            start_digit = int(part[2:])
            end_digit = int(parts[i + 1][2:])
            params['allowed_digits'] = set(str(d) for d in range(start_digit, end_digit + 1))
            i += 1  # Skip the 'de' part
        
        elif part.startswith('digits'):
            # Individual digits format: digits1_2_3
            digits_part = part[6:]  # Remove 'digits' prefix
            digit_parts = [digits_part]
            
            # Collect consecutive digit parts
            j = i + 1
            while j < len(parts) and parts[j].isdigit():
                digit_parts.append(parts[j])
                j += 1
            
            params['allowed_digits'] = set(digit_parts)
            i = j - 1  # Adjust index to account for consumed parts
        
        # Parse operations
        elif part == 'allops' or part == 'all':  # Support both new and legacy naming
            params['include_addition'] = True
            params['include_subtraction'] = True
        elif part == 'add':
            params['include_addition'] = True
            params['include_subtraction'] = False
        elif part == 'sub':
            params['include_addition'] = False
            params['include_subtraction'] = True
        
        # Parse max expressions limit
        elif part.startswith('limit'):
            params['max_expressions'] = int(part[5:])
        
        i += 1
    
    # Validation
    if params['max_value'] is None:
        raise ValueError("Could not determine max_value from filename")
    
    return params

def display_parsed_parameters(params: Dict[str, Any], filename: str):
    """Display parsed parameters in a beautiful format."""
    print(f"\n{Colors.BOLD}📋 Dataset Analysis: {Path(filename).name}{Colors.ENDC}")
    print("=" * 60)
    
    print(f"\n{Colors.CYAN}📊 Generation Parameters:{Colors.ENDC}")
    print(f"  🎯 Maximum value: {Colors.BOLD}{params['max_value']}{Colors.ENDC}")
    print(f"  🎯 Minimum value: {Colors.BOLD}{params['min_value']}{Colors.ENDC}")
    
    if params['allowed_digits'] is None:
        print(f"  🔢 Allowed digits: {Colors.GREEN}All digits (0-9){Colors.ENDC}")
    else:
        sorted_digits = sorted([int(d) for d in params['allowed_digits']])
        if len(sorted_digits) > 1 and sorted_digits == list(range(sorted_digits[0], sorted_digits[-1] + 1)):
            print(f"  🔢 Allowed digits: {Colors.GREEN}Range {sorted_digits[0]}-{sorted_digits[-1]}{Colors.ENDC}")
        else:
            digits_str = ', '.join(str(d) for d in sorted_digits)
            print(f"  🔢 Allowed digits: {Colors.GREEN}{digits_str}{Colors.ENDC}")
    
    operations = []
    if params['include_addition']:
        operations.append("➕ addition")
    if params['include_subtraction']:
        operations.append("➖ subtraction")
    print(f"  🧮 Operations: {Colors.GREEN}{' and '.join(operations)}{Colors.ENDC}")
    
    if params['max_expressions']:
        print(f"  📏 Expression limit: {Colors.WARNING}{params['max_expressions']:,}{Colors.ENDC}")
    else:
        print(f"  📏 Expression limit: {Colors.GREEN}None (all possible expressions){Colors.ENDC}")
    
    # Show file info if it exists
    if Path(filename).exists():
        file_size = Path(filename).stat().st_size
        with open(filename, 'r') as f:
            line_count = sum(1 for line in f)
        
        print(f"\n{Colors.CYAN}📁 File Information:{Colors.ENDC}")
        print(f"  📄 File size: {Colors.BOLD}{file_size:,} bytes ({file_size/1024:.1f} KB){Colors.ENDC}")
        print(f"  📝 Total expressions: {Colors.BOLD}{line_count:,}{Colors.ENDC}")
    
    # Show equivalent command line
    print(f"\n{Colors.CYAN}💻 Equivalent Command:{Colors.ENDC}")
    cmd_parts = [f"python {Path(__file__).name}"]
    
    if params['max_value'] != 100:  # 100 is default
        cmd_parts.append(f"-m {params['max_value']}")
    if params['min_value'] != 0:  # 0 is default
        cmd_parts.append(f"--min-value {params['min_value']}")
    
    if params['allowed_digits'] is not None:
        sorted_digits = sorted([int(d) for d in params['allowed_digits']])
        if len(sorted_digits) > 1 and sorted_digits == list(range(sorted_digits[0], sorted_digits[-1] + 1)):
            cmd_parts.append(f'-d "{sorted_digits[0]}-{sorted_digits[-1]}"')
        else:
            digits_str = ','.join(str(d) for d in sorted_digits)
            cmd_parts.append(f'-d "{digits_str}"')
    
    if not params['include_addition']:
        cmd_parts.append("--no-addition")
    if not params['include_subtraction']:
        cmd_parts.append("--no-subtraction")
    
    if params['max_expressions']:
        cmd_parts.append(f"--max-expressions {params['max_expressions']}")
    
    print(f"  {Colors.BOLD}{' '.join(cmd_parts)}{Colors.ENDC}")
    print()

def validate_arguments(args) -> None:
    """Validate command line arguments."""
    errors = []
    
    if args.max_value < 0:
        errors.append("max-value must be non-negative")
    
    if args.min_value < 0:
        errors.append("min-value must be non-negative")
    
    if args.min_value > args.max_value:
        errors.append("min-value cannot be greater than max-value")
    
    if args.no_addition and args.no_subtraction:
        errors.append("Cannot exclude both addition and subtraction")
    
    if args.max_expressions is not None and args.max_expressions < 1:
        errors.append("max-expressions must be positive")
    
    if errors:
        print(f"{Colors.FAIL}❌ Validation Errors:{Colors.ENDC}")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)

def print_generation_summary(args, allowed_digits: Set[str], output_path: Path):
    """Print a beautiful summary of generation parameters."""
    print(f"\n{Colors.BOLD}🚀 Generation Configuration:{Colors.ENDC}")
    print("=" * 50)
    
    print(f"  🎯 Value range: {Colors.BOLD}{args.min_value} - {args.max_value}{Colors.ENDC}")
    
    if allowed_digits:
        sorted_digits = sorted([int(d) for d in allowed_digits])
        if len(sorted_digits) > 1 and sorted_digits == list(range(sorted_digits[0], sorted_digits[-1] + 1)):
            digits_display = f"Range {sorted_digits[0]}-{sorted_digits[-1]}"
        else:
            digits_display = ', '.join(str(d) for d in sorted_digits)
        print(f"  🔢 Allowed digits: {Colors.GREEN}{digits_display}{Colors.ENDC}")
    else:
        print(f"  🔢 Allowed digits: {Colors.GREEN}All digits (0-9){Colors.ENDC}")
    
    operations = []
    if not args.no_addition:
        operations.append("➕ addition")
    if not args.no_subtraction:
        operations.append("➖ subtraction")
    print(f"  🧮 Operations: {Colors.GREEN}{' and '.join(operations)}{Colors.ENDC}")
    
    if args.max_expressions:
        print(f"  📏 Expression limit: {Colors.WARNING}{args.max_expressions:,}{Colors.ENDC}")
    else:
        print(f"  📏 Expression limit: {Colors.GREEN}Unlimited{Colors.ENDC}")
    
    print(f"  📁 Output file: {Colors.CYAN}{output_path}{Colors.ENDC}")
    print()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CalcGPT Dataset Generator - Create arithmetic expression datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m 10                                    # Generate expressions 0-10, auto-named file
  %(prog)s -o data/simple.txt -m 10                 # Custom filename, max value 10
  %(prog)s -m 50 -d "1,2,3"                        # Only use digits 1,2,3 in all numbers
  %(prog)s -m 100 -d "1-5" --min-value 1           # Digit range 1-5, minimum value 1
  %(prog)s -m 20 --no-subtraction                   # Addition only
  %(prog)s -m 1000 --max-expressions 500            # Limit to 500 expressions
  %(prog)s --analyze datasets/example.txt           # Analyze existing dataset
        """
    )
    
    # Analysis mode
    parser.add_argument(
        '--analyze',
        type=str,
        help='Analyze an existing dataset file to show generation parameters'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path (auto-generated if not specified)'
    )
    
    # Generation parameters
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
        help='Allowed digits: "1,2,3" or "1-3,7,9" (default: all digits)'
    )
    
    parser.add_argument(
        '--max-expressions',
        type=int,
        help='Maximum number of expressions to generate'
    )
    
    # Operation control
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
    
    # Utility options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with progress indicators'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CalcGPT DataGen 1.0.0'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Suppress banner output'
    )
    
    args = parser.parse_args()
    
    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()
    
    # Handle analyze mode
    if args.analyze:
        if args.verbose:
            print(f"{Colors.CYAN}🔍 Analyzing dataset file: {args.analyze}{Colors.ENDC}")
        
        try:
            params = parse_filename_parameters(args.analyze)
            display_parsed_parameters(params, args.analyze)
            return
        except ValueError as e:
            print(f"{Colors.FAIL}❌ Analysis Error: {e}{Colors.ENDC}")
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.FAIL}❌ Unexpected error during analysis: {e}{Colors.ENDC}")
            sys.exit(1)
    
    # Validate arguments
    validate_arguments(args)
    
    # Parse allowed digits
    try:
        allowed_digits = parse_digit_set(args.allowed_digits) if args.allowed_digits else None
    except ValueError as e:
        print(f"{Colors.FAIL}❌ Invalid digit specification: {e}{Colors.ENDC}")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        output_filename = generate_output_filename(
            max_value=args.max_value,
            min_value=args.min_value,
            allowed_digits=allowed_digits,
            include_addition=not args.no_addition,
            include_subtraction=not args.no_subtraction,
            max_expressions=args.max_expressions
        )
        output_path = Path(output_filename)
    else:
        output_path = args.output
    
    # Print generation summary
    if args.verbose or not args.no_banner:
        print_generation_summary(args, allowed_digits, output_path)
    
    try:
        # Generate expressions
        start_time = time.time()
        
        if args.verbose:
            print(f"{Colors.GREEN}🎬 Starting expression generation...{Colors.ENDC}")
        
        expressions = generate_expressions(
            max_value=args.max_value,
            allowed_digits=allowed_digits,
            include_addition=not args.no_addition,
            include_subtraction=not args.no_subtraction,
            min_value=args.min_value,
            verbose=args.verbose
        )
        
        # Write to file
        count = write_expressions_to_file(
            expressions,
            output_path,
            args.max_expressions,
            args.verbose
        )
        
        total_time = time.time() - start_time
        
        # Final success message
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCCESS!{Colors.ENDC}")
        print(f"  📊 Generated: {Colors.BOLD}{count:,} expressions{Colors.ENDC}")
        print(f"  📁 Saved to: {Colors.CYAN}{output_path}{Colors.ENDC}")
        print(f"  ⏱️  Total time: {Colors.BOLD}{total_time:.1f} seconds{Colors.ENDC}")
        
        if count > 0:
            rate = count / total_time
            print(f"  🚀 Generation rate: {Colors.BOLD}{rate:.0f} expressions/second{Colors.ENDC}")
        
        # Show file size
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"  💾 File size: {Colors.BOLD}{file_size:,} bytes ({file_size/1024:.1f} KB){Colors.ENDC}")
        
    except ValueError as e:
        print(f"{Colors.FAIL}❌ Generation Error: {e}{Colors.ENDC}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Generation interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()