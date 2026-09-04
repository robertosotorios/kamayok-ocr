#!/usr/bin/env bash
set -e

# Configura tu usuario de GitHub y la versión de la imagen
GITHUB_USER="TU_USUARIO_GITHUB"
IMAGE_NAME="docling-runpod"
TAG="v1"

FULL_IMAGE="ghcr.io/${GITHUB_USER}/${IMAGE_NAME}:${TAG}"
LATEST_IMAGE="ghcr.io/${GITHUB_USER}/${IMAGE_NAME}:latest"

echo "==> Iniciando sesión en ghcr.io..."
if [ -z "$GH_PAT" ]; then
  echo "Introduce tu GitHub Personal Access Token (PAT con permiso write:packages):"
  docker login ghcr.io -u "$GITHUB_USER"
else
  echo "$GH_PAT" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin
fi

echo "==> Verificando soporte Buildx..."
docker buildx create --use --name runpod-builder 2>/dev/null || docker buildx use runpod-builder

echo "==> Compilando para linux/amd64 y subiendo a GHCR..."
docker buildx build \
  --platform linux/amd64 \
  -t "$FULL_IMAGE" \
  -t "$LATEST_IMAGE" \
  --push \
  .

echo "==> ¡Subida completada con éxito!"
echo "Imagen lista para RunPod: $FULL_IMAGE"
