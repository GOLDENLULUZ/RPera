$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONUTF8 = "1"
$result = 1

try {
    $root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    Set-Location -LiteralPath $root
    Write-Host "Close the running RPera before updating."

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git was not found. Install Git for Windows: https://git-scm.com/download/win"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv was not found. Install it in PowerShell: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    }

    if (-not (Test-Path -LiteralPath (Join-Path $root ".git"))) {
        throw "This is not a Git checkout. ZIP downloads must be updated manually; see README.md."
    }
    $repository = & git rev-parse --show-toplevel
    if ($LASTEXITCODE -ne 0) {
        throw "This is not a Git checkout. ZIP downloads must be updated manually; see README.md."
    }
    $repository = [System.IO.Path]::GetFullPath($repository).TrimEnd('\')
    if (-not [string]::Equals($repository, $root.TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The project directory must be the root of its own Git checkout."
    }
    $branch = & git symbolic-ref --quiet --short HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Detached HEAD cannot be updated automatically. Switch to your normal branch first."
    }
    $upstream = & git for-each-ref '--format=%(upstream:short)' "refs/heads/$branch"
    if ($LASTEXITCODE -ne 0 -or -not $upstream) {
        throw "This branch has no upstream. Configure its remote tracking branch first."
    }
    $changes = @(& git status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect local files."
    }
    if ($changes.Count -gt 0) {
        $changes | ForEach-Object { Write-Host $_ }
        throw "Local file changes were found. Save or resolve them before updating; no files were overwritten."
    }

    Write-Host "Fetching updates for $branch from $upstream..."
    & git fetch
    if ($LASTEXITCODE -ne 0) {
        throw "Git fetch failed. Local source files have not been updated."
    }
    Write-Host "Applying a fast-forward update without overwriting ignored user files..."
    & git -c merge.autoStash=false merge --ff-only --no-overwrite-ignore $upstream
    if ($LASTEXITCODE -ne 0) {
        throw "Git update failed. No reset, stash or conflict resolution was attempted."
    }

    Write-Host "Synchronizing Python and project dependencies..."
    & uv sync --locked --no-dev
    if ($LASTEXITCODE -ne 0) {
        throw "Source update finished, but dependency setup failed. Fix the error and run start.bat again."
    }
    Write-Host "Update complete. Double-click start.bat to launch RPera."
    $result = 0
} catch {
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Write-Host ""
Write-Host "Press Enter to close this window."
[void][Console]::ReadLine()
exit $result
