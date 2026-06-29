# Execution Details

This document describes how each supported language is compiled and executed inside the sandbox. All code is executed in a temporary directory (location controlled by the `SANDBOX_TMP_DIR` environment variable, defaulting to `/tmp`) that is destroyed after execution.

---

## Common Behaviour

- Code is written to a temporary file in the temporary directory.
- Files passed via the `files` parameter are written before execution. Both absolute and relative paths are supported; relative paths are resolved from the temp directory.
- Files listed in `fetch_files` are read back (base64-encoded) after execution.
- Compilation timeouts and execution timeouts are enforced separately.
- The working directory for all commands is the temporary directory.

---

## Python

**Version:** 3.13

**Identifiers:** `python`, `pytest`

Python has two execution modes:

- **`python`** -- Executes the code directly: `python <filename>`
- **`pytest`** -- Runs the code through the pytest framework: `pytest <filename>`

Neither mode requires compilation.

The Python runtime is installed in a conda environment called `sandbox-runtime`. The default list of pip packages (approximately 80 CPU-focused packages) can be found in `runtime/python/requirements.txt`. Major packages include: numpy, scipy, pandas, sympy, scikit-learn, matplotlib, requests, Flask, SQLAlchemy, and many more.

The sandbox environment has pre-downloaded two NLTK modules (`punkt` and `stopwords`).

---

## C++

**Version:** GCC 15.2

**Identifier:** `cpp`

**Compile command:**
```
g++ -std=c++17 <filename> -o test -lcrypto -lssl -lpthread
```

**Run command:**
```
./test
```

The compile command links against OpenSSL (`-lcrypto -lssl`) and pthreads (`-lpthread`). When running in environments where these libraries are not available, the sandbox automatically detects and removes the missing flags.

---

## Java

**Version:** JDK 25

**Identifiers:** `java`, `junit`

### java mode

**Compile command:**
```
javac -cp .:javatuples-1.2.jar:<extra_jars> Main.java
```

**Run command:**
```
java -cp .:javatuples-1.2.jar:<extra_jars> -ea Main
```

`<extra_jars>` includes all `.jar` files passed via the `files` parameter. The class name in the submitted code **must** be `Main`.

### junit mode

**Compile command:**
```
javac -cp .:junit-platform-console-standalone-1.8.2.jar:junit-jupiter-api-5.11.0-javadoc.jar:<extra_jars> *.java
```

**Run command:**
```
java -jar ./junit-platform-console-standalone-1.8.2.jar \
  --class-path .:junit-platform-console-standalone-1.8.2.jar:junit-jupiter-api-5.11.0-javadoc.jar:<extra_jars> \
  --scan-class-path
```

The sandbox automatically detects the class name in the code and places it in the corresponding `.java` file. If no class name is found, the code is placed in `Main.java`.

The JUnit and JavaTuples JARs are stored in `runtime/java/`.

---

## Go

**Version:** 1.25.9

**Identifiers:** `go`, `go_test`

For both modes, the sandbox copies the built-in Go project template from `runtime/go/` to the temporary directory and writes the code there. This template includes a `go.mod` with common dependencies pre-cached.

### go mode

**Compile command:**
```
go build -o out <filename>
```

**Run command:**
```
./out
```

### go_test mode

**Run command:**
```
go test <filename>
```

No separate compilation step -- `go test` compiles and runs in one step.

---

## JavaScript (Node.js)

**Version:** Node.js 24.0.0

**Identifier:** `nodejs`

After the project environment is set up, `node_modules` is symbolically linked to the temporary directory. The code is then executed:

```
node <filename>
```

The sandbox environment includes a headless browser that can work with the puppeteer library for frontend testing.

The Node.js project configuration is in `runtime/node/` (includes package.json, babel.config.js, jest config).

---

## TypeScript

**Version:** TypeScript 5.8.3 (via tsx)

**Identifier:** `typescript`

The execution environment is identical to JavaScript above, but the entry command is:

```
tsx <filename>
```

---

## Jest

**Version:** Jest 29.7.0

**Identifier:** `jest`

The sandbox transfers the complete Node.js project (including `node_modules`, `package.json`, `babel.config.js`) to the temporary directory, writes the code to `tmpxxx.test.ts`, and runs:

```
npm run test
```

---

## C#

**Version:** .NET SDK 10.0

**Identifier:** `csharp`

The sandbox creates a new .NET console project:

```
dotnet new console -o <tmp_dir>
```

It writes the submitted code to `Program.cs`, then runs:

```
dotnet run --project <tmp_dir>
```

---

## Lean

**Version:** Lean 4.29.0

**Identifier:** `lean`

Lean code is built and run using `lake`. The Lean project is located at `runtime/lean/`.

Since Lean's Mathlib library is very slow to compile and the full compilation cache is very large, the sandbox includes a pre-compiled subset of Mathlib sufficient for MiniF2F evaluation. This subset is defined in `runtime/lean/Main.lean`.

When running code, the sandbox:

1. Copies the complete `lake` project structure to the temporary directory.
2. Writes the submitted code to `Main.lean`.
3. Runs `lake build`.

Lean's proof correctness checking happens at compile time, so there is no separate run step. The Lean language only has a `run_result` (no `compile_result`), and that corresponds to the result of the `lake build` command.

---

## PHP

**Version:** PHP CLI 8.5

**Identifier:** `php`

If the submitted code does not start with `<?php`, the sandbox automatically prepends it.

**Run command:**
```
php -f <filename>
```

---

## Scala

**Version:** Scala 3.8.3

**Identifier:** `scala`

The sandbox extracts the Scala class name from the submitted code. If no class name is found, it returns an error.

**Compile command:**
```
scalac <filename>
```

**Run command:**
```
scala <classname>
```

---

## Verilog

**Version:** Icarus Verilog 13.0

**Identifier:** `verilog`

Executed through the `evaluate_functional_correctness` approach from the [verilog-eval](https://github.com/NVlabs/verilog-eval) project.

---

## Rust

**Version:** Rust 1.95.0

**Identifier:** `rust`

**Compile command:**
```
rustc <filename> -o test
```

**Run command:**
```
./test
```

---

## Bash

**Version:** Bash 5.3.9

**Identifier:** `bash`

**Run command:**
```
/bin/bash <filename>
```

No compilation step.

---

## Lua

**Version:** Lua 5.2

**Identifier:** `lua`

**Run command:**
```
lua <filename>
```

No compilation step.

---

## R

**Version:** R 4.5.2

**Identifier:** `R`

**Run command:**
```
Rscript <filename>
```

No compilation step.

---

## Perl

**Version:** Perl 5.40.1

**Identifier:** `perl`

**Run command:**
```
perl <filename>
```

No compilation step.

---

## D

**Version:** DMD 2.112.0

**Identifier:** `D_ut`

**Compile command:**
```
dmd <filename> -unittest -of=test
```

**Run command:**
```
./test
```

The `-unittest` flag enables D's built-in unit test blocks.

---

## Ruby

**Version:** Ruby 4.0.0

**Identifier:** `ruby`

**Run command:**
```
ruby <filename>
```

No compilation step.

---

## Julia

**Version:** Julia 1.11.5

**Identifier:** `julia`

**Run command:**
```
julia <filename>
```

No compilation step.

---

## Kotlin

**Version:** Kotlin 2.1.20

**Identifier:** `kotlin_script`

**Run command:**
```
kotlin <filename>
```

Kotlin runs in script mode (no separate compilation step).

---

## Swift

**Version:** Swift 6.1.2

**Identifier:** `swift`

**Compile command:**
```
swiftc <filename> -o test.out
```

**Run command:**
```
./test
```

---

## Racket

**Version:** Racket 9.1

**Identifier:** `racket`

**Run command:**
```
racket <filename>
```

No compilation step.

---

## Summary Table

| Language | Identifier(s) | Version | Compiled | Compile Command | Run Command |
|----------|---------------|---------|----------|-----------------|-------------|
| Python | `python`, `pytest` | 3.13 | No | -- | `python <file>` / `pytest <file>` |
| C++ | `cpp` | GCC 15.2 | Yes | `g++ -std=c++17 <file> -o test -lcrypto -lssl -lpthread` | `./test` |
| Java | `java`, `junit` | JDK 25 | Yes | `javac -cp ... Main.java` | `java -cp ... -ea Main` |
| Go | `go`, `go_test` | 1.25.9 | Yes / No | `go build -o out <file>` | `./out` / `go test <file>` |
| JavaScript | `nodejs` | Node.js 24.0.0 | No | -- | `node <file>` |
| TypeScript | `typescript` | 5.8.3 | No | -- | `tsx <file>` |
| Jest | `jest` | 29.7.0 | No | -- | `npm run test` |
| C# | `csharp` | .NET 10.0 | No* | -- | `dotnet run --project <dir>` |
| Lean | `lean` | 4.29.0 | N/A | -- | `lake build` |
| PHP | `php` | 8.5 | No | -- | `php -f <file>` |
| Scala | `scala` | 3.8.3 | Yes | `scalac <file>` | `scala <class>` |
| Verilog | `verilog` | Icarus 13.0 | -- | -- | verilog-eval |
| Rust | `rust` | 1.95.0 | Yes | `rustc <file> -o test` | `./test` |
| Bash | `bash` | 5.3.9 | No | -- | `/bin/bash <file>` |
| Lua | `lua` | 5.2 | No | -- | `lua <file>` |
| R | `R` | 4.5.2 | No | -- | `Rscript <file>` |
| Perl | `perl` | 5.40.1 | No | -- | `perl <file>` |
| D | `D_ut` | DMD 2.112.0 | Yes | `dmd <file> -unittest -of=test` | `./test` |
| Ruby | `ruby` | 4.0.0 | No | -- | `ruby <file>` |
| Julia | `julia` | 1.11.5 | No | -- | `julia <file>` |
| Kotlin | `kotlin_script` | 2.1.20 | No | -- | `kotlin <file>` |
| Swift | `swift` | 6.1.2 | Yes | `swiftc <file> -o test.out` | `./test` |
| Racket | `racket` | 9.1 | No | -- | `racket <file>` |

\* C# uses `dotnet run` which handles compilation internally.
