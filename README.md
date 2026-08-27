# ExperimentMind

**Autonomous ML experiment lifecycle agent built with AWS Strands Agents SDK.**

ExperimentMind eliminates the experiment bookkeeping that consumes a data scientist's day. Drop a config. Walk away. Wake up to a ranked leaderboard and digest in your inbox.

## Architecture

Five specialized agents wired into a Strands Workflow:
Config Watcher → Validator → Job Launcher → Results Logger → Analyzer → Reporter

_(Architecture diagram coming soon)_

## Setup

```bash
git clone https://github.com/Rudra1x/experimentmind.git
cd experimentmind
pip install -r requirements.txt
cp .env.template .env
# Fill in your .env values
python -m api.main
```

## Hackathon

Built for the [Agents for Humans Hackathon](https://agentsforhumans.devpost.com/) — Professional Agents Track.