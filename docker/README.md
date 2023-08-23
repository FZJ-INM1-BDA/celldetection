# Docker

## Build
Build with context `./` (project root) and dockerfile `./docker/Dockerfile`.
```
docker build --tag "celldetection:latest" -f docker/Dockerfile .
```

For Mac Silicon:
```
docker build --platform linux/arm64 --tag "celldetection:latest" -f docker/Dockerfile 
```

With Multi-Platform Support:
```
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 --tag "celldetection:latest" -f docker/Dockerfile 
```

## Export
```
docker save celldetection | gzip -1 -c > celldetection.tar.gz
```

## Load
```
docker load -i celldetection.tar.gz
```


## Run
```
docker container run celldetection:latest /bin/bash -c "python -c 'print(2)'"
```
