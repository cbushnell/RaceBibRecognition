# PowerShell script to process multiple directories with bib recognition
# Add or remove directories in the $directories array below

# Define the list of directories to process
$directories = @(
    "C:\Users\Xapha\dev\turkey-trot-2025\Extra Start.Finish.untagged\Extra Start.Finish.untagged_0",
    "C:\Users\Xapha\dev\turkey-trot-2025\Extra Start.Finish.untagged\Extra Start.Finish.untagged_1",
    "C:\Users\Xapha\dev\turkey-trot-2025\Extra Start.Finish.untagged\Extra Start.Finish.untagged_2",
    "C:\Users\Xapha\dev\turkey-trot-2025\Extra Start.Finish.untagged\Extra Start.Finish.untagged_3"
    # Add more directories here as needed
    # "C:\path\to\another\directory",
    # "C:\path\to\yet\another\directory"
)

# Command parameters
$minFaceConfidence = 0.4
$minBibConfidence = 0.4
$bibRange = "0-10000"

# Process each directory
$totalDirectories = $directories.Count
$currentDirectory = 0

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bib Recognition Batch Processor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total directories to process: $totalDirectories`n" -ForegroundColor Yellow

foreach ($dir in $directories) {
    $currentDirectory++

    Write-Host "`n[$currentDirectory/$totalDirectories] Processing: $dir" -ForegroundColor Green
    Write-Host "----------------------------------------" -ForegroundColor Gray

    # Check if directory exists
    if (Test-Path $dir) {
        # Build and execute the command
        $command = "uv run python main.py `"$dir`" --min-face-confidence $minFaceConfidence --min-bib-confidence $minBibConfidence  --bib-range $bibRange"

        Write-Host "Command: $command`n" -ForegroundColor DarkGray

        # Execute the command
        Invoke-Expression $command

        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[SUCCESS] Completed processing: $dir" -ForegroundColor Green
        } else {
            Write-Host "`n[ERROR] Failed to process: $dir (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        }
    } else {
        Write-Host "[WARNING] Directory not found: $dir" -ForegroundColor Yellow
        Write-Host "Skipping...`n" -ForegroundColor Yellow
    }

    Write-Host "----------------------------------------`n" -ForegroundColor Gray
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Batch Processing Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Processed $currentDirectory out of $totalDirectories directories`n" -ForegroundColor Yellow
