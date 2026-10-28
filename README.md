# mkdyasserh.github.io

Personal site and blog (**Cracking AI**) of Yasser Khalafaoui. Live at <https://mkdyasserh.github.io>.

Built on [al-folio](https://github.com/alshedivat/al-folio) v1.x (Jekyll + the `al_folio_*` plugin gems), with a custom design layer on top.

## Writing and publishing

You don't need to touch Markdown files or build anything locally.

1. Open **<https://mkdyasserh.github.io/admin/>**.
2. Click **Sign In with Token** and paste a
   [fine-grained token](https://github.com/settings/personal-access-tokens/new) limited to this repository
   with **Contents: Read and write** (the sign-in screen links to the token page). The browser remembers it.
3. **Blog posts → New post**. Fill in the title, description and date, write the body, add images by dragging them in.
4. **Save**. That commits the post to `master`, and the _Deploy site_ workflow publishes it in about two minutes.

Things the editor handles for you:

| Want to…                    | Do this                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| Add a post to a series      | Pick the series and set its **part number**. The series card, page and navigation update.   |
| Start a new series          | **Series → New series**, then pick it on the posts.                                         |
| Keep a draft                | Turn **Published** off. It is saved in the repo but not shown on the site.                  |
| Schedule a post             | Set the date in the future. A daily build (06:00 UTC) publishes it on that day.             |
| Write maths                 | `$$ x^2 $$` works anywhere. Stay in Markdown mode for maths-heavy posts.                    |
| Turn off comments on a post | Untick **Allow comments**.                                                                  |

Prefer files? Posts are plain Markdown in `_posts/` (`YYYY-MM-DD-slug.md`) with front matter like:

```yaml
---
title: Attention Mechanisms Explained
description: "One or two sentences."
date: 2025-01-12 11:00:00 +0100
series: stable-diffusion # optional, the file name of a series in _series/
part: 3 # optional, order within the series
tags: [attention, transformers]
cover: assets/img/attention_meme.png # optional
---
```

## How the site is put together

| Path                               | What it is                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| `_pages/about.md`                  | Home page. Headline, intro, focus areas and bio live in the front matter / body.  |
| `assets/json/resume.json`          | Experience and education shown on the home page.                                  |
| `_bibliography/papers.bib`         | Publications. Entries with `selected={true}` appear on the home page.             |
| `_projects/`                       | Project write-ups shown as cards on the home page.                                |
| `_posts/`, `_series/`              | Blog posts and the series that group them.                                        |
| `_pages/contact.md`                | Contact page. The form backend is set under `contact_form:` in `_config.yml`.     |
| `admin/`                           | The writing desk ([Sveltia CMS](https://github.com/sveltia/sveltia-cms)).          |
| `_layouts/`, `_includes/`, `_sass/` | Local overrides of al-folio's theme files (design, blog, about, contact).         |
| `_sass/_themes.scss`               | Colour palette for light and dark mode.                                           |
| `_sass/_site.scss`                 | The design layer: typography, components and page styles.                         |
| `assets/js/site.js`                | Table of contents, blog topic filter and the contact form.                        |

## Running locally

Needs Ruby 3.x (on macOS: `brew install ruby`, then put `/opt/homebrew/opt/ruby/bin` first on your `PATH`).

```bash
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve --livereload
```

Then open <http://localhost:4000>. If the build complains about `invalid byte sequence in US-ASCII`, run `export LANG=en_US.UTF-8` first.

## Updating al-folio

Theme code now ships as gems, so most updates are `bundle update` of the `al_*` gems. To pull starter changes:

```bash
git fetch upstream
git merge upstream/main
bundle exec al-folio upgrade audit --no-fail
bundle exec al-folio upgrade overrides audit
```

The overrides audit flags local files in `_layouts/`, `_includes/` and `_sass/` whose upstream version changed.

## License

The theme is [MIT licensed](LICENSE), from al-folio. Content © Yasser Khalafaoui.
