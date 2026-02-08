# Speech Dataset Builder & Podcast Downloader

This repository contains an automated pipeline for building large-scale speech datasets from podcasts. It handles everything from fetching RSS feeds to generating speaker-diarized transcriptions using WhisperX.

## Overview

The system consists of two main components:
1. **Downloader:** Fetches audio from RSS feeds and organizes them in S3 storage.
2. **Processor:** Transcribes audio and performs speaker diarization using GPU acceleration.

## Storage Structure

The system automatically organizes data in your S3 bucket based on the `S3_BASE_PATH` configured in your `.env` file.
- `raw/`: Stores MP3 files organized by podcast name.
- `processed/`: Stores the final JSONL datasets.
- `raw/subscriptions.txt`: Tracks monitored podcasts.

## Configuration

All configuration (S3 credentials, API tokens, model settings) is managed via the `.env` file. Please ensure this file is present and correctly populated before running the scripts.

## Usage

### 1. Podcast Downloader
The downloader script is stateful and tracks your subscriptions in S3.

**Add a new podcast:**
```bash
python src/download_podcast.py --url "[https://rss-feed-url.xml](https://rss-feed-url.xml)"
