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
import concurrent.futures

try:
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"

    def _compute():
        """Helper function to compute score with parsing and verification."""
        parsed_output = parse(model_output, parsing_timeout=None)
        parsed_ground_truth = parse(ground_truth_boxed, parsing_timeout=None)
        return verify(parsed_output, parsed_ground_truth, timeout_seconds=None)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_compute)
            ret_score = future.result(timeout=10)  # 10 seconds timeout
    except concurrent.futures.TimeoutError:
        ret_score = timeout_score
    except Exception:
        pass
    return ret_score
