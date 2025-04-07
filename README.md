# Custom einops.rearrange Implementation

## Overview

This notebook implements a simplified version of the `einops.rearrange` operation for NumPy arrays. It supports:
- **Reshaping:** Including splitting and merging of axes.
- **Transposition:** Reordering axes as specified by the pattern.
- **Repeating:** Expanding a singleton axis (denoted by a literal `1` on the left) into a new axis on the right. For example, the pattern `a 1 c -> a b c` repeats the singleton axis to size `b` (provided via `axes_lengths`).

## Design Decisions

- **Pattern Parsing:**  
  The pattern string (e.g., `"a 1 c -> a b c"`) is split into groups. A literal `1` on the left is interpreted as a placeholder for a repeating axis. Ellipsis (`...`) is also supported for handling arbitrary batch dimensions.
  
- **Recipe Preparation:**  
  The `_prepare_rearrange_recipe` function validates the pattern against the input tensor’s shape, infers missing axis lengths, and determines:
  - The shape for an initial reshape (if any merging or splitting is required).
  - The axes permutation for reordering.
  - The final desired shape.
  - Which axes require repeating (i.e., where the input has size 1 but the output should be larger).

- **Repeating Handling:**  
  After initial reshape and permutation, if the expected final number of elements exceeds the current count (because of repeating singleton axes), the function uses `np.repeat` to expand those axes to the required size.

- **Error Handling:**  
  The implementation uses a custom `EinopsError` to report invalid patterns, mismatched dimensions, or missing/incorrect `axes_lengths`.

## How to Run

1. **Implementation:**  
   The first cell contains the full implementation. Ensure that the cell runs without errors.

2. **Unit Tests:**  
   The second cell includes several tests covering:
   - Transposition
   - Splitting of axes
   - Merging of axes
   - Repeating of a singleton axis
   - Handling of batch dimensions via ellipsis  
   Run this cell to execute all tests. You should see confirmation that all tests have passed.

3. **Usage Example:**  
   ```python
   import numpy as np
   x = np.random.rand(3, 1, 5)
   y = rearrange_numpy(x, 'a 1 c -> a b c', b=4)
   print(y.shape)  # Expected output: (3, 4, 5)