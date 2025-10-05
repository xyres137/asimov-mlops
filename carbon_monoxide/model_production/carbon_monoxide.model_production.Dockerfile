FROM ghcr.io/prefix-dev/pixi:0.40.0 AS build

COPY pixi.toml pixi.lock .
RUN pixi install

FROM ubuntu:22.04 AS production

RUN apt update && apt install curl -y

COPY --from=build /.pixi/envs/default /.pixi/envs/default

ENV PATH=/.pixi/envs/default/bin:$PATH
ENV CONDA_PREFIX=/.pixi/envs/default
ENV PROJ_LIB=/.pixi/envs/default/share/proj

ENTRYPOINT []
