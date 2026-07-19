$ErrorActionPreference = "Stop"

$splits = 2, 3, 4, 5

foreach ($split in $splits) {
    $resultDir = "result_paperlike\50salads_split${split}_50ep"
    $config = Join-Path $resultDir "config.yaml"
    $trainLog = Join-Path $resultDir "train.log"
    $trainErr = Join-Path $resultDir "train.err.log"

    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] split${split}: train start"
    conda run -n asrf python train.py $config 1> $trainLog 2> $trainErr
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] split${split}: train done"

    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] split${split}: evaluate boundary refinement"
    conda run -n asrf python evaluate.py $config --refinement_method refinement_with_boundary

    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] split${split}: done"
}
