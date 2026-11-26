# Qwen Image Chat のセットアップ: Python 環境、stable-diffusion.cpp、モデル一式をダウンロードします。
# 使い方: setup.bat をダブルクリック
#   -Backend vulkan|cuda12|rocm|cpu   sd.cpp のビルド (既定: vulkan。AMD/NVIDIA/Intel GPU で動作)
#   -WithBase        高品質 Base モデル (7.3GB) も取得
#   -SkipEnhancer    プロンプト強化モデル (約12GB, Ollama 必須) を取得しない
param(
    [ValidateSet("vulkan", "cuda12", "rocm", "cpu")][string]$Backend = "vulkan",
    [switch]$WithBase,
    [switch]$SkipEnhancer
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = $PSScriptRoot
Set-Location $Root

function Resolve-RootPath([string]$p, [string]$default) {
    if (-not $p) { $p = $default }
    if ([System.IO.Path]::IsPathRooted($p)) { return $p }
    return (Join-Path $Root $p)
}
$cfg = @{}
if (Test-Path "$Root\local.json") {
    $json = Get-Content "$Root\local.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($p in $json.PSObject.Properties) { $cfg[$p.Name] = $p.Value }
} else {
    Copy-Item "$Root\local.example.json" "$Root\local.json"
}
$ModelRoot = Resolve-RootPath $cfg["model_root"] "models\qwen-image-2.1"
$SdDir     = Resolve-RootPath $cfg["sdcpp_dir"] "bin\sdcpp"

function Get-File([string]$url, [string]$dest) {
    if (Test-Path $dest) {
        # resume-safe: curl -C - continues partial files and is a no-op for complete ones
    }
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Write-Host "  ↓ $(Split-Path $dest -Leaf)"
    & curl.exe -L --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 33) { throw "download failed: $url" }
}

# ---- 1) Python environment
Write-Host "[1/4] Python 環境"
$venvPy = "$Root\app\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.12 "$Root\app\.venv"
        uv pip install --python $venvPy -r "$Root\requirements.txt"
    } else {
        $py = (Get-Command python -ErrorAction SilentlyContinue)
        if (-not $py) { throw "Python 3.10+ をインストールしてください (https://www.python.org/)" }
        & python -m venv "$Root\app\.venv"
        & $venvPy -m pip install -q -r "$Root\requirements.txt"
    }
}

# ---- 2) stable-diffusion.cpp
Write-Host "[2/4] stable-diffusion.cpp ($Backend)"
if (-not (Test-Path "$SdDir\sd-server.exe")) {
    $rel = Invoke-RestMethod "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
    $pattern = switch ($Backend) {
        "vulkan" { "bin-win-vulkan-x64.zip" } "cuda12" { "bin-win-cuda12-x64.zip" }
        "rocm" { "bin-win-rocm" } "cpu" { "bin-win-cpu-x64.zip" }
    }
    $asset = $rel.assets | Where-Object { $_.name -like "*$pattern*" } | Select-Object -First 1
    if (-not $asset) { throw "release asset not found for $Backend" }
    $zip = "$env:TEMP\$($asset.name)"
    Get-File $asset.browser_download_url $zip
    New-Item -ItemType Directory -Force $SdDir | Out-Null
    Expand-Archive -Force $zip $SdDir
    if ($Backend -eq "cuda12") {
        $rt = $rel.assets | Where-Object { $_.name -like "cudart-*" } | Select-Object -First 1
        if ($rt) { Get-File $rt.browser_download_url "$env:TEMP\$($rt.name)"; Expand-Archive -Force "$env:TEMP\$($rt.name)" $SdDir }
    }
    Write-Host "  sd.cpp $($rel.tag_name) -> $SdDir"
}

# ---- 3) Qwen-Image-2.1 models (~14GB)
Write-Host "[3/4] Qwen-Image-2.1 モデル"
$HF = "https://huggingface.co"
$C = "$HF/Comfy-Org/Qwen-Image-2.1/resolve/main"
Get-File "$C/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors" "$ModelRoot\diffusion_models\qwen_image_2.1_turbo_int8_convrot.safetensors"
Get-File "$C/vae/qwen_image_2.1_vae_bf16.safetensors" "$ModelRoot\vae\qwen_image_2.1_vae_bf16.safetensors"
Get-File "$HF/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf" "$ModelRoot\text_encoders\Qwen3VL-8B-Instruct-Q4_K_M.gguf"
Get-File "$HF/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-F16.gguf" "$ModelRoot\text_encoders\mmproj-Qwen3VL-8B-Instruct-F16.gguf"
if ($WithBase) {
    Get-File "$C/diffusion_models/qwen_image_2.1_int8_convrot.safetensors" "$ModelRoot\diffusion_models\qwen_image_2.1_int8_convrot.safetensors"
}

# ---- 4) official prompt enhancers (Ollama)
Write-Host "[4/4] プロンプト強化モデル (Qwen-Image-2.1-PE)"
if ($SkipEnhancer) {
    Write-Host "  スキップしました"
} elseif (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "  Ollama が見つからないためスキップ (https://ollama.com/ を入れてから再実行すると有効になります)"
} else {
    $PE = "$ModelRoot\prompt_enhancer"
    $P = "$HF/prithivMLmods"
    Get-File "$HF/Qwen/Qwen-Image-2.1-PE-T2I/resolve/main/system_prompt.txt" "$PE\sys_T2I.txt"
    Get-File "$HF/Qwen/Qwen-Image-2.1-PE-I2I/resolve/main/system_prompt.txt" "$PE\sys_I2I.txt"
    Get-File "$P/Qwen-Image-2.1-PE-T2I-GGUF/resolve/main/Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf" "$PE\Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf"
    Get-File "$P/Qwen-Image-2.1-PE-I2I-GGUF/resolve/main/Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf" "$PE\Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf"
    Get-File "$P/Qwen-Image-2.1-PE-I2I-GGUF/resolve/main/Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf" "$PE\Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf"
    $params = "PARAMETER temperature 1.0`nPARAMETER top_p 0.95`nPARAMETER top_k 20`n"
    Set-Content -Encoding ASCII "$PE\Modelfile.t2i" ("FROM ./Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf`n" + $params)
    Set-Content -Encoding ASCII "$PE\Modelfile.i2i" ("FROM ./Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf`nFROM ./Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf`n" + $params)
    Push-Location $PE
    ollama create qwen-image-pe-t2i -f Modelfile.t2i
    ollama create qwen-image-pe-i2i -f Modelfile.i2i
    Pop-Location
}

Write-Host ""
Write-Host "セットアップ完了。start.bat で起動します。"
