# Entity Encoding Method from Job-SDF Benchmark

## Overview
The Job-SDF benchmark uses a multi-dimensional entity encoding scheme to combine multiple entity IDs into single integers.

## Encoding Scheme

### 1. **Multi-Dimensional Encoding** (for composite entities)
For entities composed of multiple dimensions (e.g., company + region), the encoding formula is:

```
encoded_id = sum([entity[i] * col_num[i-1] if i > 0 else entity[i] for i in enumerate])
```

Where:
- `entity[i]` is the i-th dimension (e.g., company_id, region_id)
- `col_num[i-1]` is the number of unique values in dimension i-1
- The first dimension is not multiplied; subsequent dimensions are multiplied by the cardinality of previous dimensions

**Example (for company + region):**
```
encoded_id = company_id + region_id * num_companies
```

### 2. **Skill ID Offset**
Skills are offset by a constant value (200000) to distinguish them from other entity types:

```
encoded_skill_id = original_skill_id + 200000
```

### 3. **Spatial Resolution (r0, r1, r2)**
The benchmark defines different granularity levels:
- **r0**: 28 skills (low frequency)
- **r1**: 8,966 skills (medium frequency)  
- **r2**: 59,921 skills (high frequency)

## Usage in Benchmark

From `predygae/data_process.ipynb`:

```python
# Create column index for entities
col_index = [f"{i}_id" for i in data_name.split('-')]  # e.g., ['company_id', 'region_id']

# Get cardinality of each dimension
col_num = [df[col].nunique() for col in col_index]

# Encode multi-dimensional entity IDs
df[data_name] = df.apply(
    lambda x: int(sum([
        x[v]*col_num[i-1] if i > 0 else x[v] 
        for i, v in enumerate(col_index)
    ])), 
    axis=1
)

# Offset skill IDs
df['skill_id'] = df['skill_id'] + 200000
```

## Decoding Process

To reverse the encoding:

```python
def decode_entity(encoded_id, dimensions):
    """
    Decode a multi-dimensional entity ID
    
    Args:
        encoded_id: The encoded ID
        dimensions: List of dimension cardinalities [dim0_count, dim1_count, ...]
    
    Returns:
        List of original IDs for each dimension
    """
    decoded = []
    remaining = encoded_id
    
    for i in range(len(dimensions) - 1, -1, -1):
        if i == 0:
            decoded.insert(0, remaining)
        else:
            divisor = 1
            for j in range(i):
                divisor *= dimensions[j]
            
            value = remaining // divisor
            decoded.insert(0, value)
            remaining = remaining % divisor
    
    return decoded
```

## Entity Type Identification

From the entities dictionary in predygae data:
- Node IDs < 30,000: Occupation/Company/Region entities
- Node IDs >= 30,000: Skill entities (offset by 200,000)

```python
# Determine entity types
pos_num_nodes = (entities[1] < 30000).sum()  # Non-skill entities
skill_num_nodes = (entities[1] >= 30000).sum()  # Skill entities
```

## References
- Source: `benchmark/predygae/data_process.ipynb`
- Implementation: `benchmark/predygae/code/mf.py`, `train_task1.py`, `train_task2.py`
