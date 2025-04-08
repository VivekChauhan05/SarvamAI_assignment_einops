# NumPy Rearrange Implementation

This project provides a NumPy-based implementation of the `rearrange` function, inspired by the `einops` library. The `rearrange_numpy` function allows users to flexibly reshape, transpose, split, and merge axes of NumPy arrays using a simple pattern string. This functionality is useful for tasks such as image processing, data manipulation, and preparing inputs for machine learning models. The code is implemented in a Google Colab notebook, making it easy to run and experiment with directly in the cloud.

## Approach

The implementation of `rearrange_numpy` is broken down into three main components:

1. **Pattern Parsing** (`_parse_expression`):
   - This function parses one side of the pattern string (e.g., `'b c (h w)'`) into its structural components.
   - It identifies axes names, detects the presence of an ellipsis (`...`), and validates the syntax.
   - The output includes a composition list (e.g., `[['b'], ['c'], ['h', 'w']]`) and a set of unique identifiers.

2. **Recipe Preparation** (`_prepare_rearrange_recipe`):
   - This function takes the pattern string (e.g., `'b h w c -> b c h w'`), the input shape, and any provided `axes_lengths` to create a "recipe" for the rearrangement.
   - The recipe consists of three steps:
     - **Initial Reshape**: Decomposes dimensions if parentheses are used on the left side (e.g., splitting `(h w)` into `h` and `w`).
     - **Axes Permutation**: Determines the order to transpose axes to match the right side.
     - **Final Reshape**: Merges dimensions if parentheses are used on the right side (e.g., merging `h w` into `(h w)`).
   - It handles ellipsis expansion and validates shape compatibility.

3. **Execution** (`rearrange_numpy`):
   - Applies the recipe to the input NumPy array by performing the necessary `reshape` and `transpose` operations.
   - Ensures element count consistency at each step to prevent invalid operations.

The approach is inspired by the `einops` library but is tailored to work exclusively with NumPy arrays within a Colab environment, avoiding external dependencies beyond NumPy itself.

## Design Decisions

Several design choices were made to ensure functionality, usability, and robustness:

- **Custom Error Handling**:
  - A custom `EinopsError` class is defined to provide specific, informative error messages for issues like invalid patterns, shape mismatches, or missing `axes_lengths`.

- **Ellipsis Support**:
  - The implementation supports ellipsis (`...`) to handle variable-length batch dimensions, replacing it with temporary axis names (e.g., `_ellipsis_0`) during processing.

- **Axes Lengths**:
  - The function accepts `axes_lengths` as keyword arguments to specify sizes for decomposition (e.g., splitting a dimension into `h` and `w`). Lengths can be inferred when possible, enhancing flexibility.

- **Validation**:
  - Extensive checks ensure:
    - Patterns include a `'->'` separator.
    - Axes names are valid and consistent between input and output.
    - Input shape matches the pattern, with proper handling of dimensions of size 1.
    - `axes_lengths` are positive integers and consistent with the input.

- **Test Suite**:
  - A comprehensive `run_tests` function includes test cases for:
    - Simple reordering (e.g., transposing).
    - Composition (e.g., merging axes).
    - Decomposition (e.g., splitting axes).
    - Identity transformations.
    - Adding/removing dimensions of size 1.
    - Ellipsis usage.
    - Edge cases (e.g., empty arrays, zero-sized dimensions).
    - Error conditions (e.g., invalid patterns, shape mismatches).


## How to Run the Code

1. **Run the Code**:
   - Step 1: **Download the above Collab file**.
   - Step 2: **Run the Code**:
   To run the code, simply execute the cell containing the **Implementation Code**, **Fuction Calling the Test Cases** and then calling the all the **test cases** step by step.
2. **Experiment**:
   - Modify the examples in the `__main__` block or add new cells to test custom patterns and arrays.
   - For example, try:
     ```python
     import numpy as np
     x = np.random.rand(5, 10, 15)
     result = rearrange_numpy(x, 'b h w -> b (h w)')
     print(result.shape)  # Should print (5, 150)