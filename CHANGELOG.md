# CHANGELOG


## v0.8.0 (2026-01-17)

### Features

- Add explanation text for selected theme in picker
  ([`d41018f`](https://github.com/gezibash/ghostty-ambient/commit/d41018fe092dd89c1610be17ccfdfac9c0245808))

Shows why the selected theme is recommended based on its characteristics (brightness, warmth) and
  match quality (score, confidence, learning phase).


## v0.7.0 (2026-01-16)

### Features

- Use Rich Live display to eliminate theme picker flickering
  ([`c7440f9`](https://github.com/gezibash/ghostty-ambient/commit/c7440f9d462169b6716b7f494fd830e7e1747103))

Replace console.clear() + render() loop with Live(screen=True) which handles the alternate screen
  buffer properly and only updates on state changes.


## v0.6.1 (2026-01-16)

### Bug Fixes

- Add RAM and disk usage to daemon status
  ([`81e2a24`](https://github.com/gezibash/ghostty-ambient/commit/81e2a249bcba939e7b20448d9fedd6b925b2e54d))


## v0.6.0 (2026-01-16)

### Features

- Show version in daemon status
  ([`f081b0a`](https://github.com/gezibash/ghostty-ambient/commit/f081b0a5c9b4e4896acc81030e9c39e28aeea8a7))


## v0.5.0 (2026-01-16)

### Continuous Integration

- Add build step before publish
  ([`9c9fb42`](https://github.com/gezibash/ghostty-ambient/commit/9c9fb42d0e412b3f5b6885a211ab46cf9674552b))

### Features

- Trigger v0.5.0 release
  ([`7c76ad4`](https://github.com/gezibash/ghostty-ambient/commit/7c76ad4f53f64250a2d3b5d6fb65a9420be33816))


## v0.4.0 (2026-01-16)

### Bug Fixes

- Disable build_command in semantic-release (we build separately)
  ([`967f769`](https://github.com/gezibash/ghostty-ambient/commit/967f769e4f7d01b6180ea6bd450c92e494e6e7f4))

- Remove invalid build_command config
  ([`68e07e9`](https://github.com/gezibash/ghostty-ambient/commit/68e07e9893e546c4377b27593e27197a408804a1))

### Continuous Integration

- Use official python-semantic-release action
  ([`7a31bc4`](https://github.com/gezibash/ghostty-ambient/commit/7a31bc45041db3d5779d36aa4351351f3e9fb7c9))

- Use semantic-release for auto-versioning on merge to main
  ([`a183a67`](https://github.com/gezibash/ghostty-ambient/commit/a183a6752f02d5bbc867ee7e75b910bfa39f1f08))

### Features

- Add --version flag to CLI
  ([`b2511a9`](https://github.com/gezibash/ghostty-ambient/commit/b2511a9df844dd582a1da77e9e9b4f8d38114466))


## v0.3.0 (2026-01-16)

### Chores

- Add tooling and release workflow
  ([`b47eaf8`](https://github.com/gezibash/ghostty-ambient/commit/b47eaf8f5370506bd668226e9a33bd44e4476c0c))

- Add ruff linting/formatting with pre-commit hooks - Add gitlint for conventional commit
  enforcement - Add python-semantic-release for versioning - Add GitHub Actions release workflow -
  Bump version to 0.3.0


## v0.1.0 (2026-01-15)
