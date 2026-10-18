param(
  [string]$OutputName = "TimbreForge-Portable-0.6.0a1.zip"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dist = Join-Path $Root "dist"
$Stage = Join-Path $Dist "TimbreForge-Portable"
$Archive = Join-Path $Dist $OutputName

python (Join-Path $Root "tools\release_check.py")
if ($LASTEXITCODE -ne 0) { throw "release_check.py failed" }

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

$Items = @(
  "timbreforge", "assets", "docs", "README.md", "CHANGELOG.md", "LICENSE",
  "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md", "MANIFEST.in",
  "pyproject.toml", "backend-lock.json", "qwen-backend-lock.json",
  "planner-backend-lock.json", "timbreforge.example.toml", "启动声线工作室.cmd",
  "安装并启动声线工作室.cmd", "tools"
)
foreach ($Item in $Items) {
  $Source = Join-Path $Root $Item
  if (Test-Path -LiteralPath $Source) { Copy-Item -LiteralPath $Source -Destination $Stage -Recurse -Force }
}

foreach ($Private in @(".local-data", ".git", ".dsh", ".codex", ".agents", ".venv", "venv")) {
  $Path = Join-Path $Stage $Private
  if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
}

Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Archive -CompressionLevel Optimal
Remove-Item -LiteralPath $Stage -Recurse -Force
Write-Host "Created $Archive"

