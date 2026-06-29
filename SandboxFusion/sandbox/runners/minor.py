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
"""Runner functions for the 12 "minor" supported languages.

Each runner follows the same pattern as the major runners: create a temp
directory, restore supplementary files, write source code, and delegate to
:func:`~sandbox.runners.base.run_commands`.  Languages that require compilation
(D, Scala, Verilog, Swift) include a compile command; interpreted languages
pass ``None``.

``MINOR_RUNNERS`` maps language identifier strings to their runner coroutines
and is merged into ``CODE_RUNNERS`` by :mod:`sandbox.runners`.
"""

import os
import re
import tempfile
from typing import Optional

from sandbox.runners.base import restore_files, run_commands
from sandbox.runners.major import _runtime_dir
from sandbox.runners.types import CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.utils.execution import get_tmp_dir


def find_scala_classname(code) -> Optional[str]:
    """Extract the Scala ``object`` name from source code using a regex.

    Args:
        code: Scala source code string.

    Returns:
        The name of the first ``object`` declaration found, or ``None`` if
        no match is found.
    """
    pat = r"object\s+(\w+)(\s*)|(\s+extends\s+\w+\s*){"
    m = re.findall(pat, code, re.DOTALL | re.MULTILINE)
    name = m[0][0] if m else None
    return name


async def run_lua(args: CodeRunArgs) -> CodeRunResult:
    """Run Lua code via the ``lua`` interpreter."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.lua', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'lua {f.name}', tmp_dir, {}, args)


async def run_r(args: CodeRunArgs) -> CodeRunResult:
    """Run R code via ``Rscript``."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.R', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'Rscript {f.name}', tmp_dir, {}, args)


async def run_perl(args: CodeRunArgs) -> CodeRunResult:
    """Run Perl code via the ``perl`` interpreter."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.pl', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'perl {f.name}', tmp_dir, {}, args)


async def run_d_ut(args: CodeRunArgs) -> CodeRunResult:
    """Compile D code with ``dmd -unittest`` and run the resulting test binary."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.d', delete=False) as f:
            f.write(args.code)

        return await run_commands(f'dmd {f.name} -unittest -of=test', './test', tmp_dir, {}, args)


async def run_ruby(args: CodeRunArgs) -> CodeRunResult:
    """Run Ruby code via the ``ruby`` interpreter."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.rb', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'ruby {f.name}', tmp_dir, {}, args)


async def run_scala(args: CodeRunArgs) -> CodeRunResult:
    """Compile Scala code with ``scalac`` and run the detected object.

    Returns an error result immediately if no ``object`` name can be extracted
    from the source code.
    """
    classname = find_scala_classname(args.code)
    if not classname:
        result = CommandRunResult(status=CommandRunStatus.Error, return_code=1, stderr='Object name not found.')
        return CodeRunResult(compile_result=result)

    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.scala', delete=False) as f:
            f.write(args.code)

        # Scala 3.x `scala` is now scala-cli and no longer supports `scala ClassName`.
        # Use `java -cp` with Scala library jars from the coursier cache instead.
        scala_cp = (
            'SCALA_CP=$('
            'find /root/.cache/coursier/v1 -name "scala-library-*.jar" -not -name "*sources*" '
            '2>/dev/null | sort -V | tail -1);'
            'SCALA3_CP=$('
            'find /root/.cache/coursier/v1 -name "scala3-library_3-*.jar" -not -name "*sources*" '
            '2>/dev/null | sort -V | tail -1);'
        )
        run_cmd = f'{scala_cp} java -cp ".:$SCALA_CP:$SCALA3_CP" {classname}'
        return await run_commands(f'scalac {f.name}', run_cmd, tmp_dir, {}, args)


async def run_julia(args: CodeRunArgs) -> CodeRunResult:
    """Run Julia code via the ``julia`` interpreter."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.jl', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'julia {f.name}', tmp_dir, {}, args)


async def run_kotlin_script(args: CodeRunArgs) -> CodeRunResult:
    """Run Kotlin script code via the ``kotlin`` command."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.kts', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'kotlin {f.name}', tmp_dir, {}, args)


async def run_verilog(args: CodeRunArgs) -> CodeRunResult:
    """Compile SystemVerilog code with ``iverilog`` (IEEE 1800-2012) and simulate with ``vvp``."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.sv', delete=False) as f:
            f.write(args.code)
            # Refer to https://github.com/NVlabs/verilog-eval/blob/4fa0ac4ed70ff1685114c25cd2e4c17cbba6a0c4/verilog_eval/execution.py#L82
            compile_cmd = f"iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o test.vvp {f.name};"
            run_cmd = f"vvp -n test.vvp"
        return await run_commands(compile_cmd, run_cmd, tmp_dir, {}, args)


async def run_lean(args: CodeRunArgs) -> CodeRunResult:
    """Build Lean 4 code via ``lake build``, symlinking shared Lake project files.

    PID isolation is disabled because ``lake`` spawns sub-processes that need
    to share the PID namespace.  The build step itself verifies that all
    theorems are proven; there is no separate execution step.
    """
    deps_dir = _runtime_dir('lean')
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        os.makedirs(os.path.join(tmp_dir, '.lake'))
        for fn in ['.lake/packages', 'lake-manifest.json', 'lakefile.lean', 'lean-toolchain']:
            os.symlink(os.path.join(deps_dir, fn), os.path.join(tmp_dir, fn))
        with open(os.path.join(tmp_dir, 'Main.lean'), 'w') as f:
            f.write(args.code)
        '''
        for lean project, `lake build` ensures that all theorems are proven, there is no need
        to execute the (possibly without entrypoint) generated binary.
        '''
        return await run_commands(None, f'lake build', tmp_dir, {}, args, disable_pid_isolation=True)


async def run_swift(args: CodeRunArgs) -> CodeRunResult:
    """Compile Swift code with ``swiftc`` and run the resulting binary."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.swift', delete=False) as f:
            f.write(args.code)
        return await run_commands(f'swiftc {f.name} -o test.out', './test.out', tmp_dir, {}, args)


async def run_racket(args: CodeRunArgs) -> CodeRunResult:
    """Run Racket code via the ``racket`` interpreter."""
    with tempfile.TemporaryDirectory(dir=get_tmp_dir(), ignore_cleanup_errors=True) as tmp_dir:
        restore_files(tmp_dir, args.files)
        with tempfile.NamedTemporaryFile(mode='w', dir=tmp_dir, suffix='.rkt', delete=False) as f:
            f.write(args.code)

        return await run_commands(None, f'racket {f.name}', tmp_dir, {}, args)


# Mapping of language identifier strings to their async runner coroutines.
MINOR_RUNNERS = {
    'lua': run_lua,
    'R': run_r,
    'perl': run_perl,
    'D_ut': run_d_ut,
    'ruby': run_ruby,
    'scala': run_scala,
    'julia': run_julia,
    'kotlin_script': run_kotlin_script,
    'verilog': run_verilog,
    'lean': run_lean,
    'swift': run_swift,
    'racket': run_racket,
}
