# PowerShell script to split a directory into subdirectories with N files each
# The subdirectories will be named: <dirname>_0, <dirname>_1, <dirname>_2, etc.
# Can also re-split existing subdirectories into a different distribution
# Can also combine files from subdirectories back into the main directory

# ============ CONFIGURATION ============
# Target directory containing files to split
$targetDirectory = "C:\Users\Xapha\dev\turkey-trot-2025\Extra Start.Finish.untagged"

# Number of files per subdirectory (only used in split/resplit modes)
$filesPerDirectory = 200

# File extensions to include (leave empty array to include all files)
$includeExtensions = @(".jpg", ".jpeg", ".png", ".tiff", ".tif")
# To include all files, use: $includeExtensions = @()

# Operation mode: "split", "resplit", or "combine"
# - "split": Split files in directory into subdirectories
# - "resplit": Re-distribute files from existing subdirectories into new distribution
# - "combine": Move all files from subdirectories back into main directory
$operationMode = "combine"
# =======================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Directory Splitter/Combiner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Validate target directory exists
if (-not (Test-Path $targetDirectory)) {
    Write-Host "[ERROR] Target directory not found: $targetDirectory" -ForegroundColor Red
    exit 1
}

# Get the directory name (without path)
$dirName = Split-Path $targetDirectory -Leaf

# ========== COMBINE MODE ==========
if ($operationMode -eq "combine") {
    Write-Host "`nMode: COMBINE subdirectories into main directory" -ForegroundColor Magenta
    Write-Host "Scanning for subdirectories in: $targetDirectory" -ForegroundColor Yellow

    # Get all subdirectories
    $subdirectories = Get-ChildItem -Path $targetDirectory -Directory

    if ($subdirectories.Count -eq 0) {
        Write-Host "[WARNING] No subdirectories found to combine" -ForegroundColor Yellow
        exit 0
    }

    Write-Host "Found $($subdirectories.Count) subdirectories" -ForegroundColor Green

    # Collect all files from all subdirectories
    $allFiles = @()
    foreach ($subdir in $subdirectories) {
        if ($includeExtensions.Count -gt 0) {
            $subdirFiles = Get-ChildItem -Path $subdir.FullName -File | Where-Object {
                $includeExtensions -contains $_.Extension.ToLower()
            }
        } else {
            $subdirFiles = Get-ChildItem -Path $subdir.FullName -File
        }
        $allFiles += $subdirFiles
        Write-Host "  $($subdir.Name): $($subdirFiles.Count) files" -ForegroundColor Gray
    }

    $totalFiles = $allFiles.Count

    if ($totalFiles -eq 0) {
        Write-Host "[WARNING] No files found in subdirectories" -ForegroundColor Yellow
        exit 0
    }

    if ($includeExtensions.Count -gt 0) {
        Write-Host "File filter: $($includeExtensions -join ', ')" -ForegroundColor Gray
    } else {
        Write-Host "File filter: All files" -ForegroundColor Gray
    }

    Write-Host "Total files to combine: $totalFiles" -ForegroundColor Green

    # Confirm before proceeding
    Write-Host "`nThis will move all files from subdirectories back into: $targetDirectory" -ForegroundColor Yellow
    Write-Host "Empty subdirectories will be removed after files are moved." -ForegroundColor Yellow
    $confirmation = Read-Host "Do you want to proceed? (y/n)"

    if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
        Write-Host "`nOperation cancelled by user." -ForegroundColor Yellow
        exit 0
    }

    Write-Host "`nCombining files into main directory...`n" -ForegroundColor Cyan

    # Track subdirectories for later cleanup
    $subdirPaths = $subdirectories | ForEach-Object { $_.FullName }
    $filesMoved = 0

    # Move all files to main directory
    foreach ($file in $allFiles) {
        $destinationPath = Join-Path $targetDirectory $file.Name

        # Check for name conflicts
        if (Test-Path $destinationPath) {
            Write-Host "[WARNING] File already exists in target directory: $($file.Name)" -ForegroundColor Yellow
            Write-Host "  Skipping to avoid overwrite..." -ForegroundColor Yellow
            continue
        }

        try {
            Move-Item -Path $file.FullName -Destination $destinationPath -Force
            $filesMoved++

            # Show progress every 50 files
            if ($filesMoved % 50 -eq 0) {
                $percentComplete = [Math]::Round(($filesMoved / $totalFiles) * 100, 1)
                Write-Host "[PROGRESS] Moved $filesMoved/$totalFiles files ($percentComplete%)" -ForegroundColor Gray
            }
        } catch {
            Write-Host "[ERROR] Failed to move file: $($file.Name)" -ForegroundColor Red
            Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    # Clean up empty subdirectories
    Write-Host "`nCleaning up empty subdirectories..." -ForegroundColor Cyan
    $subdirectoriesRemoved = 0

    foreach ($subdirPath in $subdirPaths) {
        if (Test-Path $subdirPath) {
            # Check if directory is empty
            $remainingItems = Get-ChildItem -Path $subdirPath
            if ($remainingItems.Count -eq 0) {
                try {
                    Remove-Item -Path $subdirPath -Force -Recurse
                    $subdirName = Split-Path $subdirPath -Leaf
                    Write-Host "[REMOVED] Deleted empty subdirectory: $subdirName" -ForegroundColor Yellow
                    $subdirectoriesRemoved++
                } catch {
                    Write-Host "[WARNING] Could not remove directory: $subdirPath" -ForegroundColor Yellow
                }
            } else {
                $subdirName = Split-Path $subdirPath -Leaf
                Write-Host "[WARNING] Subdirectory not empty, keeping: $subdirName ($($remainingItems.Count) items remain)" -ForegroundColor Yellow
            }
        }
    }

    # Summary
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Combine Complete!" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Total files moved: $filesMoved" -ForegroundColor Green
    Write-Host "Subdirectories removed: $subdirectoriesRemoved" -ForegroundColor Green
    Write-Host "`nDone!`n" -ForegroundColor Green
    exit 0
}

# ========== SPLIT/RESPLIT MODES ==========
# Determine mode and collect files
if ($operationMode -eq "resplit") {
    Write-Host "`nMode: RE-SPLIT existing subdirectories" -ForegroundColor Magenta
    Write-Host "Scanning for subdirectories in: $targetDirectory" -ForegroundColor Yellow

    # Get all subdirectories
    $subdirectories = Get-ChildItem -Path $targetDirectory -Directory

    if ($subdirectories.Count -eq 0) {
        Write-Host "[WARNING] No subdirectories found to re-split" -ForegroundColor Yellow
        exit 0
    }

    Write-Host "Found $($subdirectories.Count) subdirectories" -ForegroundColor Green

    # Collect all files from all subdirectories
    $allFiles = @()
    foreach ($subdir in $subdirectories) {
        if ($includeExtensions.Count -gt 0) {
            $subdirFiles = Get-ChildItem -Path $subdir.FullName -File | Where-Object {
                $includeExtensions -contains $_.Extension.ToLower()
            }
        } else {
            $subdirFiles = Get-ChildItem -Path $subdir.FullName -File
        }
        $allFiles += $subdirFiles
        Write-Host "  $($subdir.Name): $($subdirFiles.Count) files" -ForegroundColor Gray
    }

} else {
    Write-Host "`nMode: SPLIT files in directory" -ForegroundColor Cyan
    Write-Host "Scanning directory: $targetDirectory" -ForegroundColor Yellow

    # Get all files directly in the directory (not in subdirectories)
    if ($includeExtensions.Count -gt 0) {
        $allFiles = Get-ChildItem -Path $targetDirectory -File | Where-Object {
            $includeExtensions -contains $_.Extension.ToLower()
        }
    } else {
        $allFiles = Get-ChildItem -Path $targetDirectory -File
    }
}

if ($includeExtensions.Count -gt 0) {
    Write-Host "File filter: $($includeExtensions -join ', ')" -ForegroundColor Gray
} else {
    Write-Host "File filter: All files" -ForegroundColor Gray
}

$totalFiles = $allFiles.Count

if ($totalFiles -eq 0) {
    Write-Host "[WARNING] No files found to process" -ForegroundColor Yellow
    exit 0
}

Write-Host "Total files found: $totalFiles" -ForegroundColor Green
Write-Host "Files per subdirectory: $filesPerDirectory" -ForegroundColor Green

# Calculate number of subdirectories needed
$numSubdirectories = [Math]::Ceiling($totalFiles / $filesPerDirectory)
Write-Host "Subdirectories to create: $numSubdirectories`n" -ForegroundColor Green

# Confirm before proceeding
if ($operationMode -eq "resplit") {
    Write-Host "This will RE-DISTRIBUTE files from existing subdirectories into $numSubdirectories new subdirectories." -ForegroundColor Yellow
    Write-Host "Old subdirectories will be removed after files are moved." -ForegroundColor Yellow
} else {
    Write-Host "This will create $numSubdirectories subdirectories and move $totalFiles files." -ForegroundColor Yellow
}

$confirmation = Read-Host "Do you want to proceed? (y/n)"

if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "`nOperation cancelled by user." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nStarting file distribution...`n" -ForegroundColor Cyan

# In re-split mode, track old subdirectories to remove later
$oldSubdirectories = @()
if ($operationMode -eq "resplit") {
    $oldSubdirectories = Get-ChildItem -Path $targetDirectory -Directory | ForEach-Object { $_.FullName }
}

# Process files
$currentFileIndex = 0
$currentDirIndex = 0
$filesMoved = 0

foreach ($file in $allFiles) {
    # Determine which subdirectory this file belongs to
    $subdirIndex = [Math]::Floor($currentFileIndex / $filesPerDirectory)

    # Create subdirectory name
    $subdirName = "${dirName}_${subdirIndex}"
    $subdirPath = Join-Path $targetDirectory $subdirName

    # Create subdirectory if it doesn't exist
    if (-not (Test-Path $subdirPath)) {
        New-Item -Path $subdirPath -ItemType Directory -Force | Out-Null
        Write-Host "[CREATE] Created subdirectory: $subdirName" -ForegroundColor Green
        $currentDirIndex = $subdirIndex
    }

    # Move file to subdirectory
    $destinationPath = Join-Path $subdirPath $file.Name

    try {
        Move-Item -Path $file.FullName -Destination $destinationPath -Force
        $filesMoved++

        # Show progress every 50 files
        if ($filesMoved % 50 -eq 0) {
            $percentComplete = [Math]::Round(($filesMoved / $totalFiles) * 100, 1)
            Write-Host "[PROGRESS] Moved $filesMoved/$totalFiles files ($percentComplete%)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "[ERROR] Failed to move file: $($file.Name)" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    }

    $currentFileIndex++
}

# In re-split mode, remove old empty subdirectories
if ($operationMode -eq "resplit") {
    Write-Host "`nCleaning up old subdirectories..." -ForegroundColor Cyan
    foreach ($oldSubdir in $oldSubdirectories) {
        if (Test-Path $oldSubdir) {
            # Check if directory is empty
            $remainingFiles = Get-ChildItem -Path $oldSubdir -File
            if ($remainingFiles.Count -eq 0) {
                try {
                    Remove-Item -Path $oldSubdir -Force -Recurse
                    $subdirName = Split-Path $oldSubdir -Leaf
                    Write-Host "[REMOVED] Deleted empty subdirectory: $subdirName" -ForegroundColor Yellow
                } catch {
                    Write-Host "[WARNING] Could not remove directory: $oldSubdir" -ForegroundColor Yellow
                }
            } else {
                $subdirName = Split-Path $oldSubdir -Leaf
                Write-Host "[WARNING] Subdirectory not empty, keeping: $subdirName ($($remainingFiles.Count) files remain)" -ForegroundColor Yellow
            }
        }
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Split Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total files moved: $filesMoved" -ForegroundColor Green
Write-Host "Subdirectories created: $($currentDirIndex + 1)" -ForegroundColor Green

# Show file distribution
Write-Host "`nFile Distribution:" -ForegroundColor Yellow
for ($i = 0; $i -le $currentDirIndex; $i++) {
    $subdirName = "${dirName}_${i}"
    $subdirPath = Join-Path $targetDirectory $subdirName

    if (Test-Path $subdirPath) {
        $fileCount = (Get-ChildItem -Path $subdirPath -File).Count
        Write-Host "  $subdirName : $fileCount files" -ForegroundColor Gray
    }
}

Write-Host "`nDone!`n" -ForegroundColor Green
