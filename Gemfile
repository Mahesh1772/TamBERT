source "https://rubygems.org"

# The github-pages gem pins Jekyll and every plugin to the exact versions GitHub Pages runs, so a
# local build behaves like the deployed one. Installing plain `jekyll` instead would let you use
# features and plugin versions the deployed site silently ignores -- which is a slow way to debug.
gem "github-pages", group: :jekyll_plugins

# Ruby 3 dropped webrick from the stdlib, but `jekyll serve` still needs it for the local server.
gem "webrick", "~> 1.8"
