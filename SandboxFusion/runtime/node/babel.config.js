/**
 * Babel configuration for the SandboxFusion Node.js runtime.
 *
 * This configuration is consumed primarily by Jest (via babel-jest) to
 * transpile TypeScript test files and modern JavaScript before execution.
 *
 * Presets:
 *   - @babel/preset-env : Compiles modern ES syntax down to the currently
 *     running Node.js version, enabling use of the latest language features
 *     without a separate build step.
 *   - @babel/preset-typescript : Strips TypeScript type annotations so that
 *     .ts/.tsx files can be executed directly by Jest without requiring the
 *     full TypeScript compiler (tsc).
 */
module.exports = {
    presets: [
        ['@babel/preset-env', { targets: { node: 'current' } }], 
        '@babel/preset-typescript'
    ],
};
