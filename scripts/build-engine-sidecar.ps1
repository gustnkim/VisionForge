$ErrorActionPreference = "Stop"

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    $fallback = Join-Path $env:ProgramFiles "nodejs\node.exe"
    if (-not (Test-Path -LiteralPath $fallback)) {
        throw "Node.js was not found."
    }
    $nodePath = $fallback
}
else {
    $nodePath = $node.Source
}

& $nodePath (Join-Path $PSScriptRoot "build-engine-sidecar.mjs")
if ($LASTEXITCODE -ne 0) {
    throw "Sidecar build failed with exit code $LASTEXITCODE"
}
