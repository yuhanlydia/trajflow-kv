#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  openjdk-17-jdk-headless wget curl git ripgrep unzip bzip2 libdrm2 libxkbcommon0 libgbm1 \
  libasound2 libnss3 libxcursor1 libpulse0 libxshmfence1 \
  libdbus-glib-1-2 ffmpeg xvfb tzdata

mkdir -p /opt/android/cmdline-tools /root/.android
if [[ ! -x /opt/android/cmdline-tools/latest/bin/sdkmanager ]]; then
  wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip \
    -O /tmp/android-cli.zip
  unzip -q /tmp/android-cli.zip -d /opt/android/cmdline-tools
  mv /opt/android/cmdline-tools/cmdline-tools /opt/android/cmdline-tools/latest
fi

yes | /opt/android/cmdline-tools/latest/bin/sdkmanager --licenses >/dev/null 2>&1 || true
/opt/android/cmdline-tools/latest/bin/sdkmanager \
  'platform-tools' 'emulator' 'platforms;android-33' 'build-tools;33.0.0' \
  'system-images;android-33;google_apis;x86_64'

if ! /opt/android/emulator/emulator -list-avds | rg -q '^Pixel_6_API_33$'; then
  ANDROID_SDK_ROOT=/opt/android ANDROID_HOME=/opt/android \
    /opt/android/cmdline-tools/latest/bin/avdmanager create avd --force \
    --name Pixel_6_API_33 --device pixel_6 \
    --package 'system-images;android-33;google_apis;x86_64' <<< no
fi

if [[ ! -x /root/.local/bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
/root/.local/bin/uv python install 3.11

if [[ ! -d /root/android_world/.git ]]; then
  git clone https://github.com/google-research/android_world.git /root/android_world
fi
androidworld_revision=3e50888527ef9f29b9157ecd537e408008bb1c85
if [[ $(git -C /root/android_world rev-parse HEAD) != "$androidworld_revision" ]]; then
  git -C /root/android_world checkout "$androidworld_revision"
fi
if git -C /root/android_world apply --check /root/trajflow-kv/patches/androidworld-tcg.patch; then
  git -C /root/android_world apply /root/trajflow-kv/patches/androidworld-tcg.patch
elif git -C /root/android_world apply --reverse --check /root/trajflow-kv/patches/androidworld-tcg.patch; then
  echo 'AndroidWorld TCG patch already applied.'
else
  echo 'AndroidWorld worktree conflicts with the required TCG patch.' >&2
  exit 1
fi

/root/.local/bin/uv venv --python 3.11 /root/android_world/.venv
/root/.local/bin/uv pip install --python /root/android_world/.venv/bin/python /root/android_world
/root/.local/bin/uv pip install --python /root/android_world/.venv/bin/python pillow
echo 'AndroidWorld software-emulation environment installed.'
