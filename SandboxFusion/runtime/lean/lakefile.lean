-- ==========================================================================
-- lakefile.lean -- Lake build configuration for the SandboxFusion Lean 4
-- sandbox runtime.
--
-- This lakefile defines the "sandbox" package and its default executable
-- target whose entry point is `Main.lean`. It pulls in Mathlib4 (the
-- community mathematics library) from Git as a dependency so that
-- user-submitted Lean 4 proofs and programs can freely import any
-- Mathlib module without additional setup.
--
-- Package options:
--   * pp.unicode.fun  = true  : pretty-print `fun a => b` as `fun a ↦ b`
--   * pp.proofs.withType = false : omit type annotations in proof terms
-- ==========================================================================

import Lake
open Lake DSL

package «sandbox» where
  -- add package configuration options here
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩, -- pretty-prints `fun a ↦ b`
    ⟨`pp.proofs.withType, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «Sandbox» where
  -- add library configuration options here

@[default_target]
lean_exe «sandbox» where
  root := `Main
