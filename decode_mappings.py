import pandas as pd
import requests
import json
from pathlib import Path
import numpy as np

def download_mapping_file(filename):
    """Download a mapping file from the GitHub repository."""
    url = f"https://raw.githubusercontent.com/Job-SDF/benchmark/main/dataset/low_frequency_index/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print(f"Failed to download {filename}")
        return None

def create_decoding_mappings():
    """Create decoding dictionaries for all entity types."""

    # Download mapping files
    company_ids = download_mapping_file("company.json")
    region_ids = download_mapping_file("region.json")

    # Download skill category mappings (r0.json, r1.json, etc.)
    skill_mappings = {}
    for i in range(10):  # Try r0 through r9
        skill_ids = download_mapping_file(f"r{i}.json")
        if skill_ids:
            skill_mappings[f"r{i}"] = skill_ids
        else:
            break

    # Create decoding dictionaries
    # Note: Since the original names are hidden for privacy,
    # we'll create mappings from encoded IDs (0,1,2,...) to the actual IDs from the JSON files

    decoding_maps = {
        'company_id': {i: company_ids[i] for i in range(len(company_ids))} if company_ids else {},
        'region_id': {i: region_ids[i] for i in range(len(region_ids))} if region_ids else {},
    }

    # Add skill mappings for each category
    for category, skill_ids in skill_mappings.items():
        decoding_maps[f'{category}_id'] = {i: skill_ids[i] for i in range(len(skill_ids))}

    return decoding_maps

def decode_parquet_file(parquet_path, decoding_maps, output_path=None):
    """Decode a parquet file using the mapping dictionaries."""

    # Read the parquet file
    df = pd.read_parquet(parquet_path, engine='fastparquet')

    # Decode each column if mapping exists
    decoded_df = df.copy()

    for col in df.columns:
        if col in decoding_maps:
            print(f"Decoding column: {col}")
            mapping = decoding_maps[col]
            decoded_df[col] = df[col].map(mapping)
        else:
            print(f"No mapping found for column: {col}")

    # Save decoded file if output path provided
    if output_path:
        decoded_df.to_parquet(output_path, engine='fastparquet')
        print(f"Saved decoded file to: {output_path}")

    return decoded_df

def inspect_decoding(parquet_path, decoding_maps, sample_size=5):
    """Inspect the decoding by showing original vs decoded values."""

    df = pd.read_parquet(parquet_path, engine='fastparquet')

    print(f"\nInspecting file: {parquet_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Show sample of original values
    print(f"\nOriginal values (first {sample_size} rows):")
    print(df.head(sample_size))

    # Show decoded values for columns that have mappings
    decoded_sample = df.head(sample_size).copy()
    for col in df.columns:
        if col in decoding_maps:
            mapping = decoding_maps[col]
            decoded_sample[col] = decoded_sample[col].map(mapping)

    print(f"\nDecoded values (first {sample_size} rows):")
    print(decoded_sample)

    return df

# Example usage
if __name__ == "__main__":
    # Create decoding mappings
    print("Downloading mapping files...")
    decoding_maps = create_decoding_mappings()

    print("Available mappings:")
    for key, mapping in decoding_maps.items():
        print(f"  {key}: {len(mapping)} entries")

    # Example: Decode one of the parquet files
    parquet_files = list(Path("datasets").glob("*.parquet"))
    if parquet_files:
        example_file = parquet_files[0]
        print(f"\nDecoding example file: {example_file}")

        # Inspect the decoding
        inspect_decoding(example_file, decoding_maps)

        # Optionally save decoded version
        # output_path = example_file.parent / f"decoded_{example_file.name}"
        # decode_parquet_file(example_file, decoding_maps, output_path)