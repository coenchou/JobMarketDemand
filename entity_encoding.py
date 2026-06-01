"""
Entity Encoding/Decoding module based on Job-SDF benchmark

This module provides encoding and decoding functions for multi-dimensional
entity IDs used in the Job-SDF dataset.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union


class EntityEncoder:
    """Encodes and decodes multi-dimensional entity IDs."""
    
    SKILL_OFFSET = 200000  # Offset for skill IDs to separate from other entities
    
    def __init__(self, dimensions: Dict[str, int]):
        """
        Initialize the encoder with dimension cardinalities.
        
        Args:
            dimensions: Dictionary mapping dimension names to cardinalities
                       e.g., {'company': 521, 'region': 1609, 'skill': 2324}
        """
        self.dimensions = dimensions
        self.dim_names = list(dimensions.keys())
        self.dim_counts = [dimensions[d] for d in self.dim_names]
    
    def encode(self, entity_dict: Dict[str, int]) -> int:
        """
        Encode a multi-dimensional entity into a single integer.
        
        Args:
            entity_dict: Dictionary with dimension names as keys
                        e.g., {'company': 5, 'region': 10, 'skill': 100}
        
        Returns:
            Encoded integer ID
        """
        encoded = 0
        multiplier = 1
        
        # Process dimensions in order
        for i, dim_name in enumerate(self.dim_names):
            if dim_name in entity_dict:
                if i == 0:
                    encoded += entity_dict[dim_name]
                else:
                    encoded += entity_dict[dim_name] * multiplier
                    multiplier *= self.dimensions[dim_name]
            else:
                raise ValueError(f"Missing dimension: {dim_name}")
        
        # Apply skill offset if present
        if 'skill' in entity_dict:
            encoded += self.SKILL_OFFSET
        
        return encoded
    
    def decode(self, encoded_id: int) -> Dict[str, int]:
        """
        Decode a multi-dimensional entity ID into component dimensions.
        
        Args:
            encoded_id: The encoded entity ID
        
        Returns:
            Dictionary with decoded dimension values
        """
        # Remove skill offset if present
        has_skill = encoded_id >= self.SKILL_OFFSET
        if has_skill:
            encoded_id -= self.SKILL_OFFSET
        
        decoded = {}
        remaining = encoded_id
        
        # Decode from last dimension to first
        for i in range(len(self.dim_names) - 1, -1, -1):
            dim_name = self.dim_names[i]
            
            if i == 0:
                decoded[dim_name] = remaining
            else:
                # Calculate divisor (product of cardinalities before this dimension)
                divisor = 1
                for j in range(i):
                    divisor *= self.dim_counts[j]
                
                value = remaining // divisor
                decoded[dim_name] = value
                remaining = remaining % divisor
        
        # Reorder dictionary by original dim_names order
        decoded = {k: decoded[k] for k in self.dim_names}
        
        return decoded
    
    def encode_dataframe(self, df: pd.DataFrame, 
                        dimension_cols: List[str],
                        output_col: str = None) -> pd.DataFrame:
        """
        Encode multiple rows from a dataframe.
        
        Args:
            df: Input dataframe
            dimension_cols: List of column names that form the entity
            output_col: Name of output column (default: join of input columns)
        
        Returns:
            Dataframe with encoded column added
        """
        if output_col is None:
            output_col = '_'.join(dimension_cols)
        
        def encode_row(row):
            entity_dict = {col: row[col] for col in dimension_cols}
            return self.encode(entity_dict)
        
        df_copy = df.copy()
        df_copy[output_col] = df_copy.apply(encode_row, axis=1)
        
        return df_copy
    
    def decode_dataframe(self, df: pd.DataFrame, 
                        encoded_col: str,
                        output_cols: List[str] = None) -> pd.DataFrame:
        """
        Decode encoded IDs in a dataframe.
        
        Args:
            df: Input dataframe with encoded column
            encoded_col: Name of column with encoded IDs
            output_cols: Names for output columns (default: uses dimension names)
        
        Returns:
            Dataframe with decoded columns added
        """
        if output_cols is None:
            output_cols = self.dim_names
        
        def decode_row(encoded_id):
            return pd.Series(self.decode(encoded_id))
        
        df_copy = df.copy()
        decoded = df_copy[encoded_col].apply(decode_row)
        decoded.columns = output_cols
        
        return pd.concat([df_copy, decoded], axis=1)


def encode_multi_dimensional_entity(entity_values: List[int], 
                                   cardinalities: List[int]) -> int:
    """
    Encode multiple entity dimensions into a single integer.
    
    Uses the formula: encoded = entity[0] + entity[1]*card[0] + entity[2]*card[0]*card[1] + ...
    
    Args:
        entity_values: List of dimension values [e.g., company_id, region_id]
        cardinalities: List of dimension cardinalities [e.g., num_companies, num_regions]
    
    Returns:
        Encoded integer ID
    
    Example:
        >>> encode_multi_dimensional_entity([5, 10], [100, 50])
        505  # = 5 + 10*100
    """
    encoded = 0
    multiplier = 1
    
    for i, value in enumerate(entity_values):
        if i == 0:
            encoded += value
        else:
            encoded += value * multiplier
        
        if i < len(cardinalities):
            multiplier *= cardinalities[i]
    
    return encoded


def decode_multi_dimensional_entity(encoded_id: int, 
                                   cardinalities: List[int]) -> List[int]:
    """
    Decode a multi-dimensional entity ID into component dimensions.
    
    Args:
        encoded_id: The encoded entity ID
        cardinalities: List of dimension cardinalities
    
    Returns:
        List of decoded dimension values
    
    Example:
        >>> decode_multi_dimensional_entity(505, [100, 50])
        [5, 10]
    """
    decoded = []
    remaining = encoded_id
    
    for i in range(len(cardinalities) - 1, -1, -1):
        if i == 0:
            decoded.insert(0, remaining)
        else:
            # Calculate divisor (product of cardinalities up to this point)
            divisor = 1
            for j in range(i):
                divisor *= cardinalities[j]
            
            value = remaining // divisor
            decoded.insert(0, value)
            remaining = remaining % divisor
    
    return decoded


def apply_skill_offset(skill_ids: Union[int, np.ndarray, pd.Series], 
                       offset: int = 200000) -> Union[int, np.ndarray, pd.Series]:
    """
    Apply skill ID offset to distinguish skills from other entities.
    
    Args:
        skill_ids: Skill ID(s) to offset
        offset: Offset value (default: 200000)
    
    Returns:
        Offset skill ID(s) with same type as input
    """
    return skill_ids + offset


def remove_skill_offset(offset_skill_ids: Union[int, np.ndarray, pd.Series], 
                       offset: int = 200000) -> Union[int, np.ndarray, pd.Series]:
    """
    Remove skill ID offset to retrieve original skill IDs.
    
    Args:
        offset_skill_ids: Offset skill ID(s)
        offset: Offset value (default: 200000)
    
    Returns:
        Original skill ID(s) with same type as input
    """
    return offset_skill_ids - offset


# Example usage
if __name__ == "__main__":
    # Example 1: Simple multi-dimensional encoding
    print("=" * 60)
    print("Example 1: Multi-dimensional Encoding")
    print("=" * 60)
    
    entity_values = [5, 10]  # company_id=5, region_id=10
    cardinalities = [100, 50]  # 100 companies, 50 regions
    
    encoded = encode_multi_dimensional_entity(entity_values, cardinalities)
    print(f"Encoded: {encoded}")
    
    decoded = decode_multi_dimensional_entity(encoded, cardinalities)
    print(f"Decoded: {decoded}")
    
    # Example 2: Using EntityEncoder class
    print("\n" + "=" * 60)
    print("Example 2: EntityEncoder Class")
    print("=" * 60)
    
    encoder = EntityEncoder({
        'company': 521,
        'region': 1609,
        'skill': 2324
    })
    
    # Encode a company entity
    company_entity = {'company': 10, 'region': 5, 'skill': 100}
    encoded_company = encoder.encode(company_entity)
    print(f"Encoded company entity: {encoded_company}")
    
    decoded_company = encoder.decode(encoded_company)
    print(f"Decoded company entity: {decoded_company}")
    
    # Example 3: Encoding with skill offset
    print("\n" + "=" * 60)
    print("Example 3: Skill Offset")
    print("=" * 60)
    
    original_skill_id = 42
    offset_skill_id = apply_skill_offset(original_skill_id)
    print(f"Original skill ID: {original_skill_id}")
    print(f"Offset skill ID: {offset_skill_id}")
    
    recovered_skill_id = remove_skill_offset(offset_skill_id)
    print(f"Recovered skill ID: {recovered_skill_id}")
