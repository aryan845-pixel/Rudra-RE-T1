# Downloads free, open-source pretrained super-resolution models into .\models
# for use with OpenCV's dnn_superres. Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File .\get_models.ps1
#
# NOTE: written without being able to test the downloads (the sandbox this project was built in
# blocks GitHub). The repository names are the ones OpenCV's dnn_superres documentation points to;
# if a URL fails, open the repository page printed below and download the file by hand.
# File names must stay exactly as below: <MODEL>_x<scale>.pb  (e.g. EDSR_x4.pb).

$ErrorActionPreference = "Continue"
$dest = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$files = @(
    @{ Name = "FSRCNN_x2.pb"; Url = "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb"; Page = "https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models" },
    @{ Name = "FSRCNN_x4.pb"; Url = "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb"; Page = "https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models" },
    @{ Name = "ESPCN_x2.pb";  Url = "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x2.pb";        Page = "https://github.com/fannymonori/TF-ESPCN/tree/master/export" },
    @{ Name = "ESPCN_x4.pb";  Url = "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x4.pb";        Page = "https://github.com/fannymonori/TF-ESPCN/tree/master/export" },
    @{ Name = "EDSR_x2.pb";   Url = "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x2.pb";       Page = "https://github.com/Saafke/EDSR_Tensorflow/tree/master/models" },
    @{ Name = "EDSR_x4.pb";   Url = "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb";       Page = "https://github.com/Saafke/EDSR_Tensorflow/tree/master/models" }
)

foreach ($f in $files) {
    $target = Join-Path $dest $f.Name
    if (Test-Path $target) { Write-Host "skip  $($f.Name) (already present)"; continue }
    try {
        Invoke-WebRequest -Uri $f.Url -OutFile $target -UseBasicParsing
        $size = (Get-Item $target).Length
        if ($size -lt 10240) { throw "downloaded file is only $size bytes (not a model)" }
        Write-Host ("ok    {0}  {1:N0} KiB" -f $f.Name, ($size / 1KB))
    } catch {
        if (Test-Path $target) { Remove-Item $target -Force }
        Write-Host "FAIL  $($f.Name): $($_.Exception.Message)"
        Write-Host "      download it manually from $($f.Page)"
    }
}
Write-Host "`nVerify with:  python .\sr_workbench.py --scale 4 --require-ai"
