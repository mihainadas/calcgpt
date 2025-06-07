"""
CalcGPT Dataset Generation Library

Core dataset generation functionality for CalcGPT arithmetic expression datasets.
"""

import time
from pathlib import Path
from typing import List, Set, Generator, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DatagenConfig:
    """Configuration for dataset generation"""
    max_value: int = 100
    min_value: int = 0
    allowed_digits: Set[str] = None
    include_addition: bool = True
    include_subtraction: bool = True
    max_expressions: int = None
    output_dir: str = "datasets"


def contains_only_allowed_digits(number: int, allowed_digits: Set[str]) -> bool:
    """Check if a number contains only the allowed digits."""
    if allowed_digits is None:
        return True
    return all(digit in allowed_digits for digit in str(number))


def generate_valid_numbers(max_value: int, allowed_digits: Set[str] = None, min_value: int = 0) -> List[int]:
    """Generate list of valid numbers based on constraints."""
    if allowed_digits is None:
        numbers = list(range(min_value, max_value + 1))
        return numbers
    
    valid_numbers = []
    for i in range(min_value, max_value + 1):
        if contains_only_allowed_digits(i, allowed_digits):
            valid_numbers.append(i)
    
    return valid_numbers


def generate_expressions(config: DatagenConfig) -> Generator[str, None, None]:
    """Generate arithmetic expressions based on configuration."""
    valid_numbers = generate_valid_numbers(
        config.max_value, 
        config.allowed_digits, 
        config.min_value
    )
    
    if not valid_numbers:
        raise ValueError("No valid numbers found with the given constraints")
    
    for i in valid_numbers:
        for j in valid_numbers:
            # Addition expressions
            if config.include_addition:
                result = i + j
                # Check if result also satisfies digit constraints
                if contains_only_allowed_digits(result, config.allowed_digits):
                    yield f"{i}+{j}={result}"
            
            # Subtraction expressions (only when i >= j to avoid negative results)
            if config.include_subtraction and i >= j:
                result = i - j
                # Check if result also satisfies digit constraints
                if contains_only_allowed_digits(result, config.allowed_digits):
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


def generate_output_filename(config: DatagenConfig) -> str:
    """Generate descriptive filename based on configuration."""
    parts = ["ds-calcgpt"]
    
    # Add min and max values
    parts.append(f"min{config.min_value}")
    parts.append(f"max{config.max_value}")
    
    # Add digit constraints
    if config.allowed_digits:
        # Sort digits to ensure consistent naming
        sorted_digits = sorted([int(d) for d in config.allowed_digits])
        
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
    if config.include_addition and config.include_subtraction:
        parts.append("allops")
    elif config.include_addition:
        parts.append("add")
    elif config.include_subtraction:
        parts.append("sub")
    
    # Add max expressions if specified
    if config.max_expressions:
        parts.append(f"limit{config.max_expressions}")
    
    filename = "_".join(parts) + ".txt"
    return str(Path(config.output_dir) / filename)


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
        elif part == 'allops' or part == 'all':
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


def get_file_stats(filepath: Path) -> Dict[str, Any]:
    """Get statistics about a dataset file."""
    if not filepath.exists():
        return None
    
    file_size = filepath.stat().st_size
    with open(filepath, 'r') as f:
        line_count = sum(1 for line in f)
    
    return {
        'file_size': file_size,
        'line_count': line_count,
        'file_size_kb': file_size / 1024
    }


class DatasetGenerator:
    """Main dataset generation class"""
    
    def __init__(self, config: DatagenConfig, verbose: bool = True):
        """Initialize generator with configuration."""
        self.config = config
        self.verbose = verbose
        
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def estimate_expressions(self) -> int:
        """Estimate the number of expressions that will be generated."""
        valid_numbers = generate_valid_numbers(
            self.config.max_value,
            self.config.allowed_digits,
            self.config.min_value
        )
        
        if not valid_numbers:
            return 0
        
        total_combinations = len(valid_numbers) ** 2
        
        # Rough estimate accounting for operation types and digit constraints
        if self.config.include_addition and self.config.include_subtraction:
            estimated = total_combinations * 1.5  # Subtraction reduces total due to i >= j constraint
        else:
            estimated = total_combinations
        
        return int(estimated)
    
    def generate_dataset(self, output_path: Path = None) -> Dict[str, Any]:
        """Generate complete dataset and return statistics."""
        start_time = time.time()
        
        # Generate output path if not provided
        if output_path is None:
            output_filename = generate_output_filename(self.config)
            output_path = Path(output_filename)
        
        self.log(f"Generating dataset: {output_path}")
        self.log(f"Value range: {self.config.min_value} - {self.config.max_value}")
        
        if self.config.allowed_digits:
            sorted_digits = sorted([int(d) for d in self.config.allowed_digits])
            if len(sorted_digits) > 1 and sorted_digits == list(range(sorted_digits[0], sorted_digits[-1] + 1)):
                digits_display = f"Range {sorted_digits[0]}-{sorted_digits[-1]}"
            else:
                digits_display = ', '.join(str(d) for d in sorted_digits)
            self.log(f"Allowed digits: {digits_display}")
        else:
            self.log("Allowed digits: All digits (0-9)")
        
        operations = []
        if self.config.include_addition:
            operations.append("addition")
        if self.config.include_subtraction:
            operations.append("subtraction")
        self.log(f"Operations: {', '.join(operations)}")
        
        if self.config.max_expressions:
            self.log(f"Expression limit: {self.config.max_expressions:,}")
        else:
            self.log("Expression limit: Unlimited")
        
        # Estimate expressions
        estimated = self.estimate_expressions()
        if estimated > 0:
            self.log(f"Estimated expressions: ~{estimated:,}")
        
        # Generate expressions
        expressions = generate_expressions(self.config)
        
        # Write to file
        count = write_expressions_to_file(
            expressions,
            output_path,
            self.config.max_expressions
        )
        
        generation_time = time.time() - start_time
        
        # Get file statistics
        file_stats = get_file_stats(output_path)
        
        # Return generation results
        return {
            'expressions_generated': count,
            'generation_time': generation_time,
            'output_path': output_path,
            'file_stats': file_stats,
            'config': self.config,
            'estimated_expressions': estimated
        } 