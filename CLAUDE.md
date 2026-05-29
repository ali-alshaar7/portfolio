# Portfolio Blog — Agent Guide

This is a Hugo static site blog. Posts are written in Markdown and live in `content/posts/`.

## Adding a new post

Create a new `.md` file in `content/posts/`. The filename becomes the URL slug, so use kebab-case.

Example: `content/posts/my-new-post.md`

Every post needs this frontmatter at the top:

```markdown
---
title: "Your Post Title"
date: 2026-05-25
tags: ["tag1", "tag2"]
summary: "One sentence shown on the homepage listing."
---

Post body goes here.
```

- `title` — displayed as the heading on the post page and the homepage
- `date` — ISO format (YYYY-MM-DD); posts are sorted by this, newest first
- `tags` — optional list; shown as pills on the homepage and post
- `summary` — optional but recommended; shown as the excerpt on the homepage

## Markdown features

**Code blocks** with syntax highlighting:

````markdown
```python
def hello():
    return "world"
```
````

Any language identifier works: `python`, `go`, `bash`, `javascript`, `c`, `cpp`, etc.

**Inline code:** wrap in backticks — `like this`

**Images:** place image files in `static/images/` and embed with:
```markdown
![alt text](/portfolio/images/filename.png)
```

**Links:**
```markdown
[link text](https://example.com)
```

**Blockquotes:**
```markdown
> This is a blockquote.
```

## Previewing locally

```bash
hugo server
```

Then open http://localhost:1313/portfolio/

## Building for production

```bash
hugo build
```

Output goes to `public/`. This is what gets deployed to GitHub Pages.

## File structure

```
content/posts/     ← blog posts go here
static/images/     ← images referenced in posts go here
layouts/           ← HTML templates (don't touch unless redesigning)
static/css/        ← stylesheet
static/js/         ← starfield animation
hugo.toml          ← site config (title, author, social links)
```
