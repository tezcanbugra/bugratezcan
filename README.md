# bugratezcan.com

Personal blog built with [Hugo](https://gohugo.io) and the [hugo-classic](https://github.com/goodroot/hugo-classic) theme. Deployed to [GitHub Pages](https://pages.github.com).

## Local Development

```bash
# Start dev server
hugo server -D

# Create a new post
hugo new post/my-new-post.md
```

## Writing Posts

Create a `.md` file in `content/post/`:

```markdown
---
title: "Post Title"
date: '2026-03-30'
categories:
  - General
tags:
  - example
---

Your content here...
```

Add images to `static/images/` and reference them as `/images/filename.jpg`.

## Deployment

Push to `main` → GitHub Actions builds and deploys automatically.
