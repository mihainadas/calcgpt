#!/usr/bin/env python3
"""
CalcGPT Dataset Generator - CLI Interface

A command-line interface for generating arithmetic expression datasets
with configurable parameters, filtering options, and analysis capabilities.

Author: Mihai NADAS
Version: 2.0.0
"""

import argparse
import sys
import time
from pathlib import Path

from lib.dategen import (
    DatasetGenerator, 
    DatagenConfig, 
    parse_digit_set, 
    parse_filename_parameters,
    get_file_stats
)

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
║                 Dataset Generation Tool                       ║
║                         v2.0.0                                ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.ENDC}"""
    print(banner)

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

def create_config_from_args(args) -> DatagenConfig:
    """Create DatagenConfig from command line arguments."""
    # Parse allowed digits
    try:
        allowed_digits = parse_digit_set(args.allowed_digits) if args.allowed_digits else None
    except ValueError as e:
        print(f"{Colors.FAIL}❌ Invalid digit specification: {e}{Colors.ENDC}")
        sys.exit(1)
    
    return DatagenConfig(
        max_value=args.max_value,
        min_value=args.min_value,
        allowed_digits=allowed_digits,
        include_addition=not args.no_addition,
        include_subtraction=not args.no_subtraction,
        max_expressions=args.max_expressions,
        output_dir="datasets"
    )

def display_parsed_parameters(params: dict, filename: str):
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
    file_stats = get_file_stats(Path(filename))
    if file_stats:
        print(f"\n{Colors.CYAN}📁 File Information:{Colors.ENDC}")
        print(f"  📄 File size: {Colors.BOLD}{file_stats['file_size']:,} bytes ({file_stats['file_size_kb']:.1f} KB){Colors.ENDC}")
        print(f"  📝 Total expressions: {Colors.BOLD}{file_stats['line_count']:,}{Colors.ENDC}")
    
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

def print_generation_summary(config: DatagenConfig, output_path: Path):
    """Print a beautiful summary of generation parameters."""
    print(f"\n{Colors.BOLD}🚀 Generation Configuration:{Colors.ENDC}")
    print("=" * 50)
    
    print(f"  🎯 Value range: {Colors.BOLD}{config.min_value} - {config.max_value}{Colors.ENDC}")
    
    if config.allowed_digits:
        sorted_digits = sorted([int(d) for d in config.allowed_digits])
        if len(sorted_digits) > 1 and sorted_digits == list(range(sorted_digits[0], sorted_digits[-1] + 1)):
            digits_display = f"Range {sorted_digits[0]}-{sorted_digits[-1]}"
        else:
            digits_display = ', '.join(str(d) for d in sorted_digits)
        print(f"  🔢 Allowed digits: {Colors.GREEN}{digits_display}{Colors.ENDC}")
    else:
        print(f"  🔢 Allowed digits: {Colors.GREEN}All digits (0-9){Colors.ENDC}")
    
    operations = []
    if config.include_addition:
        operations.append("➕ addition")
    if config.include_subtraction:
        operations.append("➖ subtraction")
    print(f"  🧮 Operations: {Colors.GREEN}{' and '.join(operations)}{Colors.ENDC}")
    
    if config.max_expressions:
        print(f"  📏 Expression limit: {Colors.WARNING}{config.max_expressions:,}{Colors.ENDC}")
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
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CalcGPT DataGen 2.0.0'
    )
    
    args = parser.parse_args()
    
    # Print banner unless suppressed
    if not args.quiet:
        print_banner()
    
    # Handle analyze mode
    if args.analyze:
        if not args.quiet:
            print(f"{Colors.CYAN}🔍 Analyzing dataset file: {args.analyze}{Colors.ENDC}")
        
        try:
            params = parse_filename_parameters(args.analyze)
            display_parsed_parameters(params, args.analyze)
            return 0
        except ValueError as e:
            print(f"{Colors.FAIL}❌ Analysis Error: {e}{Colors.ENDC}")
            return 1
        except Exception as e:
            print(f"{Colors.FAIL}❌ Unexpected error during analysis: {e}{Colors.ENDC}")
            return 1
    
    # Validate arguments
    validate_arguments(args)
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Setup output path
    output_path = args.output
    
    # Print generation summary
    if output_path and (not args.quiet):
        print_generation_summary(config, output_path)
    
    try:
        # Initialize generator
        generator = DatasetGenerator(config, verbose=not args.quiet)
        
        # Generate dataset
        results = generator.generate_dataset(output_path)
        
        # Print final results
        if not args.quiet:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCCESS!{Colors.ENDC}")
            print(f"  📊 Generated: {Colors.BOLD}{results['expressions_generated']:,} expressions{Colors.ENDC}")
            print(f"  📁 Saved to: {Colors.CYAN}{results['output_path']}{Colors.ENDC}")
            print(f"  ⏱️  Total time: {Colors.BOLD}{results['generation_time']:.1f} seconds{Colors.ENDC}")
            
            if results['expressions_generated'] > 0:
                rate = results['expressions_generated'] / results['generation_time']
                print(f"  🚀 Generation rate: {Colors.BOLD}{rate:.0f} expressions/second{Colors.ENDC}")
            
            if results['file_stats']:
                file_stats = results['file_stats']
                print(f"  💾 File size: {Colors.BOLD}{file_stats['file_size']:,} bytes ({file_stats['file_size_kb']:.1f} KB){Colors.ENDC}")
        
        return 0
        
    except ValueError as e:
        print(f"{Colors.FAIL}❌ Generation Error: {e}{Colors.ENDC}")
        return 1
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Generation interrupted by user{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {e}{Colors.ENDC}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())