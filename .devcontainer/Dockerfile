FROM ghcr.io/astral-sh/uv:0.10.12@sha256:72ab0aeb448090480ccabb99fb5f52b0dc3c71923bffb5e2e26517a1c27b7fec AS uv
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm@sha256:7876580dc67fd460fd962f004cbeb480027e9bbc0657096f1087db11f9eaff39

ARG TYPST_VERSION=0.15.1
ARG TARGETARCH
ARG DEBIAN_FRONTEND=noninteractive

COPY --from=uv /uv /uvx /usr/local/bin/

RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        fonts-noto-cjk \
        fonts-noto-color-emoji \
        make \
        xz-utils \
    && case "${TARGETARCH}" in \
        amd64) \
            typst_target="x86_64-unknown-linux-musl"; \
            typst_sha256="a6d077d0a95eed5a2eba715b2dae06be954f624ccbf85758a03f389ded33118c" \
            ;; \
        arm64) \
            typst_target="aarch64-unknown-linux-musl"; \
            typst_sha256="5aa8d74a3d906e60ea12a66ac2f37f8eef1b14cbad7182a745e393a10c23dcee" \
            ;; \
        *) \
            echo "Unsupported architecture: ${TARGETARCH}" >&2; \
            exit 1 \
            ;; \
    esac \
    && typst_archive="typst-${typst_target}.tar.xz" \
    && curl --fail --location --show-error --silent \
        "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/${typst_archive}" \
        --output "/tmp/${typst_archive}" \
    && echo "${typst_sha256}  /tmp/${typst_archive}" | sha256sum --check --strict \
    && tar --extract --file "/tmp/${typst_archive}" --xz \
        --directory /usr/local/bin \
        --strip-components 1 \
        "typst-${typst_target}/typst" \
    && typst --version \
    && fc-cache --force \
    && rm -rf /var/lib/apt/lists/* "/tmp/${typst_archive}"
