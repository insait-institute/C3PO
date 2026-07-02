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
# from . import gsm8k, math, prime_math, prime_code

import re

from verl.utils.import_utils import deprecated


def _summarize_testcase_metadata(metadata_list):
    """Build a compact per-testcase execution summary from sandbox metadata.

    Returns a dict like ``{"testcase_1": "success", "testcase_2": "runtime_error: ..."}``.
    This is dumped with rollouts instead of the full test cases, which are large.
    For failing cases a short reason (truncated stderr / API error) is appended.
    """
    summary = {}
    if not metadata_list:
        return summary
    for i, md in enumerate(metadata_list):
        if not isinstance(md, dict):
            summary[f"testcase_{i + 1}"] = str(md)[:160]
            continue
        idx = md.get("case_index", i)
        status = md.get("status", "unknown")
        detail = ""
        if status != "success":
            for field in ("api_request_error", "stderr", "compile_stderr"):
                val = md.get(field)
                if val:
                    # collapse whitespace and truncate to keep the dump small
                    detail = ": " + " ".join(str(val).split())[:160]
                    break
        summary[f"testcase_{idx + 1}"] = f"{status}{detail}"
    return summary


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    retry_on_timeout=True,
    continuous=False,
    sandbox_fusion_timeout=10,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "dummy_task":

        def verify_format_correctness(sol: str):
            pattern = r"^<think>(?!.*<think>)(.*?)</think>\s*\\boxed\{(.*?)\}$"
            match = re.match(pattern, sol, re.DOTALL | re.MULTILINE)
            return True if match else False

        return 1 if verify_format_correctness(solution_str) else -1

    elif data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif (
        data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math500", "amc23", "minervamath", "skywork_math_hard"]
        or "aime" in data_source
    ):
        # from . import math_reward

        # res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:
        from . import mathverify_verifier

        res = mathverify_verifier.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning", "math_dapo_instruct", "math_dapo_base"]:
        from . import math_dapo

        res = math_dapo.compute_score(data_source, solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"] or "code" in data_source:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here.
            # continuous (configurable via reward_model.sandbox_fusion.continuous):
            #   False -> binary verdict, 1.0 iff every test case passes
            #            (check_correctness short-circuits on the first failure);
            #   True  -> partial credit, the fraction of test cases that pass.
            score, metadata_list = sandbox_fusion.compute_score(
                sandbox_fusion_url,
                concurrent_semaphore,
                memory_limit_mb,
                solution_str,
                ground_truth,
                continuous=continuous,
                retry_on_timeout=retry_on_timeout,
                timeout=sandbox_fusion_timeout,
            )
            # Return the score plus a compact per-testcase execution summary so
            # rollouts can be dumped without the (large) full test cases. The
            # reward managers unpack "score"; the rest become reward_extra_info.
            return {
                "score": float(score),
                "acc": float(score),
                "testcase_summary": _summarize_testcase_metadata(metadata_list),
            }
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=continuous)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)


__all__ = ["default_compute_score"]
