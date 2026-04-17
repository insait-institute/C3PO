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
import multiprocessing
import logging

from math_verify import parse, verify

class TimeoutWarningFilter(logging.Filter):
    def filter(self, record):
        if "Timeout is disabled" in record.getMessage() and "prevent code getting stuck" in record.getMessage():
            return False
        return True

logging.getLogger("math_verify.parser").addFilter(TimeoutWarningFilter())
logging.getLogger("math_verify.grader").addFilter(TimeoutWarningFilter())


def _worker_compute(model_output: str, ground_truth_boxed: str, q: multiprocessing.Queue):
    """Helper worker to compute score with parsing and verification."""
    try:
        parsed_output = parse(model_output, parsing_timeout=None)
        parsed_ground_truth = parse(ground_truth_boxed, parsing_timeout=None)
        res = verify(parsed_output, parsed_ground_truth, timeout_seconds=None)
        q.put(("success", res))
    except Exception:
        q.put(("error", 0.0))


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"

    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()

    p = ctx.Process(target=_worker_compute, args=(model_output, ground_truth_boxed, q))
    p.start()
    p.join(10)  # 10 seconds timeout

    if p.is_alive():
        p.terminate()
        p.join()
        return timeout_score

    if not q.empty():
        status, res = q.get()
        if status == "success":
            return 1.0 if res else 0.0
        else:
            return 0.0
    return 0.0
