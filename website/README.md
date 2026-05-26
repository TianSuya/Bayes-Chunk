# AnyEdit++ Project Page

Static GitHub Pages site for demonstrating AnyEdit++ Bayes-Chunk segmentation in long-form knowledge editing.

This directory is deployed from the same repository as the Bayes-Chunk codebase:

```text
https://github.com/TianSuya/Bayes-Chunk
```

After GitHub Pages is enabled for the repository, the project page will be served at:

```text
https://tiansuya.github.io/Bayes-Chunk/
```

## Local Preview

```bash
cd website
python -m http.server 8017
```

Open `http://127.0.0.1:8017/`.

## Rebuild Demo Cases

The interactive cases are derived from `data/segmented_editevery.json`.

```bash
python website/scripts/build_cases.py
```

The script writes `website/static/data/cases.json`.

## Deploy

The repository includes a GitHub Actions workflow at `.github/workflows/pages.yml`.
It uploads `website/` as the GitHub Pages artifact, so the site can live in the same repository as the code.

Repository setup:

1. Open GitHub repository settings.
2. Go to **Pages**.
3. Set **Source** to **GitHub Actions**.
4. Push to `main` or `master`, or run the workflow manually from the Actions tab.

The site is fully static and does not require a Node build step.

The current Paper link points to `https://openreview.net/pdf?id=W6qfbvysDh`.
