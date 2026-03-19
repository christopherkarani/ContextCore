# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
yarn
```

## Local Development

```bash
yarn start
```

This starts a local dev server and opens a browser window. Most edits show up without a restart.

## Build

```bash
yarn build
```

This generates static content in the `build` directory so you can serve it from any static host.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you host on GitHub Pages, this is the simplest way to build the site and push it to `gh-pages`.
