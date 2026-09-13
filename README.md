# hermes-devin-acp

A [Hermes Agent](https://hermes-agent.nousresearch.com) model provider plugin that exposes your [Devin](https://devin.ai) subscription models through Hermes, using your authenticated Devin CLI. A Devin subscription is required.

## Prerequisites

1. **Devin CLI** installed and authenticated:
   ```bash
   devin auth login
   ```

2. **Hermes Agent** v0.20.0+ (with model provider plugin support)

## Install

#### Desktop / Dashboard

Open **Settings → Plugins**, click **Install from Git**, and paste `paxytools/hermes-devin-acp` or the full Git URL:

```
https://github.com/paxytools/hermes-devin-acp.git
```

#### — or —

#### CLI

```bash
hermes plugins install paxytools/hermes-devin-acp
```

## Configure

After installing, activate the provider in Hermes:

```bash
hermes model
```

Select `devin-acp` from the provider list and choose your model. Hermes handles the config automatically — no manual `config.yaml` editing needed.
