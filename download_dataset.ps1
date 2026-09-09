$DatasetDir = "c:\Users\DELL\Downloads\OrchestraNet-master\OrchestraNet-master\data\coco"
if (-not (Test-Path $DatasetDir)) {
    New-Item -ItemType Directory -Path $DatasetDir -Force | Out-Null
}

Write-Host "Downloading COCO 2017 annotations (241MB)..."
Invoke-WebRequest -Uri "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -OutFile "$DatasetDir\annotations_trainval2017.zip"
Write-Host "Extracting annotations..."
Expand-Archive -Path "$DatasetDir\annotations_trainval2017.zip" -DestinationPath "$DatasetDir" -Force
Remove-Item "$DatasetDir\annotations_trainval2017.zip"

Write-Host "Downloading COCO 2017 Val images (778MB)..."
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/val2017.zip" -OutFile "$DatasetDir\val2017.zip"
Write-Host "Extracting Val images..."
Expand-Archive -Path "$DatasetDir\val2017.zip" -DestinationPath "$DatasetDir" -Force
Remove-Item "$DatasetDir\val2017.zip"

Write-Host "Downloading COCO 2017 Train images (18GB)... This will take a long time."
Invoke-WebRequest -Uri "http://images.cocodataset.org/zips/train2017.zip" -OutFile "$DatasetDir\train2017.zip"
Write-Host "Extracting Train images..."
Expand-Archive -Path "$DatasetDir\train2017.zip" -DestinationPath "$DatasetDir" -Force
Remove-Item "$DatasetDir\train2017.zip"

Write-Host "Download and extraction complete!"
