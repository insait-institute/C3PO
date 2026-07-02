# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utility functions for the sandbox framework.

Provides common helpers used across the codebase including random string
generation, async context manager pooling, conda environment detection,
PHP code normalization, JSON parsing, string truncation, and JSONL file loading.
"""

import json
import os
import secrets
import string
import sys
from typing import Any, Dict


def generate_random_string(length):
    """Generate a cryptographically secure random alphanumeric string.

    Uses the ``secrets`` module for secure randomness, suitable for tokens,
    identifiers, and other security-sensitive contexts.

    Args:
        length: The desired length of the generated string.

    Returns:
        A random string of the specified length composed of ASCII letters
        and digits.
    """
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string


def random_cgroup_name() -> str:
    """Generate a random 16-character lowercase name for cgroup/namespace identifiers.

    Uses ``secrets`` for cryptographic randomness. 26^16 ≈ 4.4 × 10^22
    possible values, making birthday-paradox collisions negligible even
    under very high concurrency.

    Returns:
        A 16-character string of random lowercase ASCII letters.
    """
    return ''.join(secrets.choice(string.ascii_lowercase) for _ in range(16))


def find_conda_root():
    """Locate the conda root directory by walking up from the current Python executable.

    Traverses parent directories starting from ``sys.executable`` looking for a
    directory that contains a ``condabin/`` subdirectory, which indicates the
    conda installation root.

    Returns:
        The absolute path to the conda root directory if found, or an error
        message string if the conda root could not be located or an unexpected
        error occurred.
    """
    try:
        python_executable = sys.executable
        env_root = python_executable
        current_dir = env_root

        while current_dir:
            if env_root != current_dir and os.path.exists(os.path.join(current_dir, 'condabin')):
                # This indicates we are in a Conda environment
                conda_root = current_dir
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                # We have reached the root of the filesystem
                conda_root = None
                break
            current_dir = parent_dir

        if conda_root and os.path.isdir(conda_root):
            return conda_root
        else:
            return "Conda root directory not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def ensure_php_tag_in_string(php_code: str) -> str:
    """Ensure that a PHP code string starts with the ``<?php`` tag.

    Args:
        php_code: A string containing PHP code.

    Returns:
        The PHP code string, with ``<?php`` prepended if it was missing.
    """
    php_code = php_code.lstrip()
    if not php_code.startswith('<?php'):
        php_code = '<?php\n' + php_code
    return php_code


def ensure_json(obj: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Ensure that a dictionary value at the given key is parsed from JSON.

    If ``obj[key]`` is a JSON string, it is parsed in-place into a dict.
    If it is already a dict (or other non-string type), it is returned as-is.

    Args:
        obj: The dictionary containing the value to check.
        key: The key whose value should be ensured as parsed JSON.

    Returns:
        The parsed (or already-parsed) value at ``obj[key]``.
    """
    if isinstance(obj[key], str):
        obj[key] = json.loads(obj[key], strict=False)
    return obj[key]


def truncate_str(s: str, max_length: int = 1000, placeholder: str = '...') -> str:
    """Truncate a string by keeping both ends and inserting a placeholder in the middle.

    Args:
        s: Input string.
        max_length: Maximum length limit. Defaults to 1000.
        placeholder: String used as the middle placeholder. Defaults to ``'...'``.

    Returns:
        The original string if within the limit, or a truncated version with
        the placeholder in the middle.
    """
    if not s or len(s) <= max_length:
        return s

    # Ensure at least 1 character on each side of placeholder
    if max_length < len(placeholder) + 2:
        max_length = len(placeholder) + 2

    # Calculate length to keep at beginning and end
    keep_length = (max_length - len(placeholder)) // 2

    return s[:keep_length] + placeholder + s[-keep_length:]


def load_jsonl(file_path):
    """Load a JSON Lines (.jsonl) file into a list of dictionaries.

    Each line in the file is expected to be a valid JSON object.

    Args:
        file_path: Path to the .jsonl file to read.

    Returns:
        A list of dictionaries, one per line in the file.
    """
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data
