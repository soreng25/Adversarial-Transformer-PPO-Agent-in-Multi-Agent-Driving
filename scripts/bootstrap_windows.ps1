param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$ImageName = "gpudrive-adversary:a-aa48a431"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop with the WSL2 backend is required. Native Windows is not the reference GPUDrive runtime."
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher with Python 3.11 is required to fingerprint the exact build context."
}

$PortCommit = (& py -3.11 scripts/port_identity.py commit).Trim()
$PortDirty = (& py -3.11 scripts/port_identity.py dirty).Trim()
$PortDiffSha256 = (& py -3.11 scripts/port_identity.py diff_sha256).Trim()
$PortSourceTreeSha256 = (& py -3.11 scripts/port_identity.py source_tree_sha256).Trim()

docker build `
    --platform linux/amd64 `
    --file containers/gpudrive/Dockerfile `
    --tag $ImageName `
    --build-arg "PORT_GIT_COMMIT=$PortCommit" `
    --build-arg "PORT_DIRTY=$PortDirty" `
    --build-arg "PORT_DIFF_SHA256=$PortDiffSha256" `
    --build-arg "PORT_SOURCE_TREE_SHA256=$PortSourceTreeSha256" `
    .
if ($LASTEXITCODE -ne 0) { throw "Reference image build failed." }

New-Item -ItemType Directory -Force -Path artifacts/milestone-a | Out-Null
New-Item -ItemType Directory -Force -Path .cache/gpudrive | Out-Null
$artifactPath = (Resolve-Path artifacts).Path
$cachePath = (Resolve-Path .cache/gpudrive).Path
# The native module is CUDA-linked even when the simulator uses ExecMode.CPU.
$gpuArgs = @("--gpus", "all")

docker run --rm @gpuArgs `
    --shm-size=20g `
    --volume "${artifactPath}:/workspace/port/artifacts" `
    --volume "${cachePath}:/var/cache/gpudrive" `
    $ImageName `
    python -m gpudrive_adversary doctor `
        --source /opt/gpudrive `
        --reference `
        --strict `
        --output "artifacts/milestone-a/doctor-${Device}.json"
if ($LASTEXITCODE -ne 0) { throw "Reference doctor failed." }

docker run --rm @gpuArgs `
    --shm-size=20g `
    --volume "${artifactPath}:/workspace/port/artifacts" `
    --volume "${cachePath}:/var/cache/gpudrive" `
    $ImageName `
    python -m gpudrive_adversary fresh-smoke `
        --source /opt/gpudrive `
        --device $Device `
        --output "artifacts/milestone-a/fresh-${Device}"
if ($LASTEXITCODE -ne 0) { throw "Fresh-process scene smoke failed." }
