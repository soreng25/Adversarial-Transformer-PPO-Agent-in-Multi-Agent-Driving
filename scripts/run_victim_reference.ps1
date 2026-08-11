param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$ImageName = "gpudrive-adversary:a-aa48a431"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop with the WSL2 backend and NVIDIA GPU support is required."
}

docker image inspect $ImageName *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Reference image $ImageName is missing; run scripts/bootstrap_windows.ps1 first."
}

New-Item -ItemType Directory -Force -Path artifacts/milestone-b | Out-Null
New-Item -ItemType Directory -Force -Path .cache/gpudrive | Out-Null
$artifactPath = (Resolve-Path artifacts).Path
$cachePath = (Resolve-Path .cache/gpudrive).Path
$checkpoint = "/opt/checkpoints/policy_S10_000_02_27/1532950cad84dafc6e9d976a2bcc524ee481a1a1"

docker run --rm --gpus all `
    --shm-size=20g `
    --volume "${artifactPath}:/workspace/port/artifacts" `
    --volume "${cachePath}:/var/cache/gpudrive" `
    $ImageName `
    python -m gpudrive_adversary verify-victim-checkpoint `
        --source /opt/gpudrive `
        --checkpoint $checkpoint `
        --output "artifacts/milestone-b/checkpoint-${Device}.json"
if ($LASTEXITCODE -ne 0) { throw "Pinned victim checkpoint verification failed." }

docker run --rm --gpus all `
    --shm-size=20g `
    --volume "${artifactPath}:/workspace/port/artifacts" `
    --volume "${cachePath}:/var/cache/gpudrive" `
    $ImageName `
    python -m gpudrive_adversary victim-fresh-eval `
        --source /opt/gpudrive `
        --checkpoint $checkpoint `
        --device $Device `
        --output "artifacts/milestone-b/fresh-${Device}"
if ($LASTEXITCODE -ne 0) { throw "Fresh-process deterministic victim evaluation failed." }
