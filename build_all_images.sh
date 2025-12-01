#!/usr/bin/env bash
set -euo pipefail

# Rebuild all Skeleton Action Lab Docker images with their default tags.
# Run from the repository root:
#   bash build_all_images.sh

images=(
  "skeleton-lab:baselines baselines.docker"
  "skeleton-lab:ctrgcn ctrgcn.docker"
  "skeleton-lab:msg3d msg3d.docker"
  "skeleton-lab:freqmixformer freqmixformer.docker"
  "skeleton-lab:skateformer skateformer.docker"
  "skeleton-lab:hypergcn hypergcn.docker"
  "skeleton-lab:fsvae fsvae.docker"
  "skeleton-lab:msf-gzssar msf-gzssar.docker"
)

for entry in "${images[@]}"; do
  read -r tag dockerfile <<<"$entry"
  echo "======================================"
  echo "Building $tag from $dockerfile"
  docker build -t "$tag" -f "$dockerfile" .
done

echo "All images built."
