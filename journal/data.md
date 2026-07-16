# TamilBERT v0.1 — Data

Building a monolingual Tamil corpus turned out to be a project on its own.

Tamil text is available online, but not always in a form that is directly useful for pretraining. A lot of it comes mixed with formatting artifacts, duplicated web content, English fragments, metadata, headers, page numbers, and other noise that a model does not need to learn. So the goal here was not just to gather *more* text, but to build a corpus that is reasonably large, readable, and defensible.

For this version, three sources were used:

1. [Project Madurai](https://www.projectmadurai.org/pmworks.html) — public-domain Tamil literature.
2. [CC-100 Tamil](https://data.statmt.org/cc-100/ta.txt.xz) — Tamil text extracted from Common Crawl.
3. [Tamil Wikipedia](https://dumps.wikimedia.org/tawiki/latest/tawiki-latest-pages-articles.xml.bz2) — Tamil encyclopedia-style writing from Wikipedia dumps.

These three sources were chosen because together they cover very different styles of Tamil:

- Project Madurai contributes older and more literary language.
- Tamil Wikipedia contributes formal, edited, informational prose.
- CC-100 contributes scale, even if it is noisier than the other two.

**Why a mix?** A corpus made from only one source can become stylistically narrow. A corpus made from only web crawl data can become large but repetitive and messy. This pipeline tries to keep the strengths of each source while reducing the kinds of noise that would hurt downstream training.

---

## Source overview

Before getting into the scripts, it helps to know what each source is doing in the corpus.

| Source | What it adds | Main risk |
|---|---|---|
| Project Madurai | Literary and classical Tamil, often clean and meaningful line by line | Headers, numbering, very short poetic lines, edition artifacts |
| Tamil Wikipedia | Formal modern Tamil, broad topical coverage | XML markup, templates, links, tables, collapsed formatting |
| CC-100 Tamil | Massive scale and broad web coverage | Heavy duplication, metadata leakage, mixed formatting |

For the final corpus each one needs different extraction and cleanup logic before it becomes useful.

---

## Project Madurai

Project Madurai is a digital archive of Tamil literary works. Many texts are available as HTML pages, so the approach here is to collect the relevant HTML links and extract the visible body text from each page.

### What the script does

1. Visit the main Project Madurai works page.
2. Collect links that point to UTF-8 HTML texts.
3. Open each page and read only the HTML `body`.
4. Split the body text into lines.
5. Clean each line before writing it out.

The cleaning step removes obvious web artifacts while keeping the Tamil content:

- Remove URLs.
- Remove email addresses.
- Remove HTML entities.
- Extract Tamil words from the line.
- Keep the line only if it still contains Tamil text.

The output is written to a source file for downstream processing.

### Why this approach

Only the page body is used because that is where the actual text lives in the sampled Project Madurai pages. Menus, navigation, and other page regions are not useful training data, so they are intentionally ignored.

Some lines still contain digits or numbering. That was not treated as an automatic defect. In literary sources, numbers can mark verse order, section order, or structural sequence, so removing every digit blindly would throw away useful information along with noise.

Another important decision came later in the pipeline: lines with fewer than 2 tokens are treated as invalid. That sounds strict at first, but it was chosen carefully because some literary lines are genuinely short by design. Setting the minimum too high would hurt Project Madurai more than the other sources.

---

## Tamil Wikipedia

Tamil Wikipedia is valuable because it contains large amounts of formal, modern Tamil across many topics. But raw Wikipedia dumps are not ready-to-train text. They contain XML structure, wiki markup, templates, links, headings, tables, and other formatting that must be stripped away.

### What the script does

1. Download the Tamil Wikipedia dump.
2. Parse it page by page using `lxml`.
3. Keep only the main article namespace.
4. Read the revision text for each page.
5. Clean wiki-specific formatting.
6. Split cleaned page text into smaller chunks.
7. Write normal lines to one file and very long lines to a separate file.

The cleaning step includes:

- Removing template blocks.
- Removing reference-like markup.
- Removing HTML tags and comments.
- Resolving wiki links to their display text.
- Removing bare URLs.
- Removing formatting markers such as bold and italics.
- Removing section heading markup while keeping the heading text itself.

After cleaning, the page text is split into smaller chunks before Tamil word extraction. This matters because Wikipedia pages can collapse a lot of unrelated formatting into one giant piece of text if they are handled too coarsely.

### Why the long-line split exists

This was one of the most important decisions in the pipeline.

During analysis, some Wikipedia lines were found to be extremely long. Those were not “rich long paragraphs” in the usual sense. Many of them were broken tables, formatting residue, or other flattened wiki structures that survived the first cleaning pass.

To deal with this, lines longer than 2500 tokens were separated into a dedicated file for inspection. That threshold was not meant to be mathematically perfect. It was a practical cutoff chosen because the other sources stayed well below that range, and these giant Wikipedia lines clearly behaved differently.

Later analysis showed that most of those long lines had very poor real-content density, which confirmed that the long-line split was catching structural noise rather than valuable prose.

---

## CC-100 Tamil

CC-100 Tamil is the biggest source in the pipeline by far. Its main value is scale. Without it, the corpus would be much smaller. But web-crawled text comes with trade-offs: duplication, inconsistent formatting, short boilerplate lines, and fragments that are clearly not meant as clean language data.

### What the script does

1. Download the Tamil CC-100 file.
2. Read it line by line from the compressed archive.
3. Decode each line as UTF-8.
4. Remove URLs and HTML entities.
5. Extract Tamil words from each line.
6. Keep only lines that still contain Tamil content.

Compared with Wikipedia, CC-100 does not need XML parsing or wiki-markup cleanup. The bigger issue here is not nested structure but repetition and web noise.

### Why CC-100 is still worth using

Even after cleaning, CC-100 remains the noisiest source in the project. But it is also the source that provides the overwhelming majority of the corpus volume.

That means the right question is not “is CC-100 perfect?” It is “can enough value be kept while the worst problems are controlled?” The later QA passes showed that the answer was yes, but only after adding merge-time deduplication and a content-ratio filter.

---

## Content-ratio filtering

A basic extraction pass is not enough on its own. Some lines can survive cleaning and still be low-value because they contain too little actual Tamil content relative to symbols, numbering, or formatting debris.

To catch this, an extra filtering stage was added based on `real_content_ratio`.

### What this metric means

For each line:

1. Split the line into tokens.
2. Count how many tokens contain at least one Tamil character.
3. Divide that count by the total number of tokens.

This produces a ratio between 0 and 1.

A line with a high ratio is mostly Tamil text. A line with a low ratio is often dominated by symbols, formatting fragments, metadata, broken tables, or mixed-content junk.

### Threshold used

The threshold was set to `0.3`.

This means a line is kept if at least 30% of its tokens contain Tamil characters. Otherwise it is sent to a separate “below threshold” file for audit instead of being used downstream.

This is intentionally conservative. A stricter threshold would remove more noise, but it would also risk deleting valid lines from literary sources where numbering and formatting are part of the text layout. A looser threshold would let too much structural junk through, especially from Wikipedia long lines.

### Why this was added

This filter was added after inspecting unusually long Wikipedia lines and finding that many of them were not actually meaningful text. Once that issue became visible, the same rule was applied to all sources so that the pipeline stayed consistent.

The “below threshold” files are not thrown away. They are kept for inspection because they are useful for understanding what kinds of lines are being excluded and whether the rule should change in later versions.

---

## Cleaning before merge

At this point, each source has gone through source-specific extraction and then through the shared content-ratio filter.

That produces two versions of each source:

- an `above_threshold` file, used for downstream work;
- a `below_threshold` file, kept for audit and manual review.

Only the `above_threshold` files move forward into merging.

This separation is useful for two reasons:

1. It keeps the training path simple and consistent.
2. It preserves evidence for later debugging instead of silently deleting questionable lines.

That second point matters more than it seems. Corpus work becomes very hard to trust when every discarded line disappears without explanation.

---

## Merge and deduplication

Once the cleaned source files are ready, they are merged into a single corpus file.

### What happens during merge

1. Read all `above_threshold` text files.
2. Merge them into one file.
3. Skip any line that does not contain at least one Tamil character.

This produces a single merged corpus made only from the cleaned, above-threshold inputs.

### Why deduplication happens after merge

Deduplication is done on the merged corpus, not on each source separately.

That decision matters because duplicates do not exist only *within* a source. The same sentence or line can appear in more than one source, especially when a web-crawled source overlaps with public web content derived from Wikipedia or other published text.

The deduplication step uses an exact hash of each line and keeps only the first occurrence. Any later repeats are written to a duplicate file for inspection.

Doing this after merge solves two problems at once:

- it removes duplicates inside a source;
- it also removes duplicates across different sources.

This was especially important because CC-100 contributed a very large amount of repeated content. Without a post-merge dedup step, the final corpus would have been much larger on paper but much less diverse in practice.

---

## Train and test split

After deduplication, the corpus is split into train and test files.

### What the script does

1. Count the number of lines in the deduplicated corpus.
2. Assign 90% of lines to train and 10% to test.
3. Shuffle the assignment list using a fixed random seed.
4. Write each line into either `train.txt` or `test.txt`.

The fixed seed is important because it makes the split reproducible. If the pipeline is re-run later, the logic remains stable and comparable.

The split is done after cleaning and deduplication so that both train and test are drawn from the same final-quality pool rather than from noisy intermediate files.

---

## Why some imperfect lines are still kept

Some imperfect lines were intentionally kept:

- Short literary lines with real meaning.
- Numbered lines where the numbering helps preserve structure.
- Informal punctuation that does not corrupt the text itself.

Some imperfect lines were intentionally dropped:

- Symbol-heavy lines with too little Tamil content.
- Duplicate lines that only inflate repetition.
- Very short invalid lines that are mostly headers, digits, or formatting leftovers.

This is the general rule behind the whole pipeline: aim for text that is useful for language modeling not for perfection.

---

## Final note

The main lesson from this stage of the project is that collecting data is only the first half of the work. The second half is making decisions that are simple enough to defend, consistent enough to automate, and cautious enough not to destroy valid Tamil text by over-cleaning.

That is why this pipeline keeps audit files, separates source-specific cleaning from shared filtering, and uses metrics to justify changes instead of changing rules blindly. The corpus is not “perfect,” but it is traceable, explainable, and much better suited for downstream tokenizer training and masked-language-model pretraining than the raw source dumps.