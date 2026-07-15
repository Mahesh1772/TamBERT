# TamilBERT v0.1 — Data 

Finding data for training a monolingual BERT model in Tamil was quite a challange on itself. Rich works representing clean language usage encompassing the intricasies of the language are sarce. Most prevelant data online are from tweets, news articles and wikipedia. These while containing sizeable text for training are often tainted with alot of opinions and mixed language words, poor translation, local dialects and inconsistent grammer.

The sources chosen for training the v1.0 of the tokenizer is as follows:
1. (Project Madurai)[https://www.projectmadurai.org/pmworks.html] - A digital archive of old tamil literature work
2. CC100 - A web crawl of many news articles, and tweets scrapped off the web
3. Tamil Wiki - Every tamil wikipedia page online

Apart from this others sources of data can be found in the link here. For the future versions we plan to integrate X and Y into the training for more data representative of the language and of good quality.

---

## Project Madurai

This source is a website maintained where all sources of poems and other works are available as pdfs and html pages. We use the main page to source the content from all the .html pages.

1. Collect all the links in the literature page of the site.
2. Go to each page, use `BeautifulSoup` to extract the text alone from the body. We extract and store this as a list. For each line in this a cleaning process is used.
    - strip any links in the text body
    - strip any emails (the site had email option on every page to report bugs)
    - strip english alphabets from site headers and numbers which have page number, phone number, etc.
    - Extract out tamil letters + punctuaion `[!,.-]` + digits
    - check if each line has atleast one tamil alphabet, only move to next step if it does
    - return cleaned list
3. Subsequently write each page to `project_madurai_extracted.txt` file 

### Decision made
- There are some lines with orphaned digits, for instance the number subceding sangam literature in Aathichudi. These are left in intentionally and not fully remomved as in some lines they do have importance as to their ordering.
- Only the text from the `BeautifulSoup.body` (body tag) is extracted here. This was decided after sampling a few random .html pages manually and printing their respective texts and in the small tested sample from the 1300+ links, all of them had the most important literature part of the body. Hence any and all information and text in all others parts of the page are igonred intentionally. 
- The `invalid_line_limit` a parameter explained in the later stgages of the pipeline was set to 2 (meaning all lines with lesser than 2 words will not end up in the final corpus) based on this data source as some works have only 2 words per line by design.

---

## Tamil Wikipedia

This source is an aggregate of web scrapped wikipedia pages specifically in tamil. It contains the text along with tables, formating, font info and links as well as section headings.

1. Extract the compressed file line by line and write to 2 different files depending on line length.
2. Using a XML parser `lxml` extract out only the pages from the file and not all the other metadata as they are assumed to be in english and not relevant for training. For every page as we read it the following is done
    - find the text portion of the page using specifc handles used in `lxml`. If no text found, make it null
    - strip template formatting block, html tags, links, comments, special characters, links, urls, formattings, tables, sections headings
    - Split page text into smaller chunks to filter out unwated chars and only retain tamil alphabets+punctuation+digits
    - check if each line has atleast one tamil alphabet, only move to next step if it does
    - check the length of the currently existing line read from file, store it to different files according to length.
    - clear memory after writing to file as this will expand to be a 12GB file.
3. Data written to `tamil_wiki_extracted.txt` and `tamil_wiki_long_lines_extracted.txt` file 

### Decision made
- The idea to split by pages only occured after analysis described in `data_stats.md`. Few pages had huge chuncks of text upto 70k. On review these happened to be broken table and header formatting with very little text. To confirm the text amount, the ration of text to symbols was run and found out only lesser than 0.30 of text was actually meaningful within the lines contained distorted formatting rather than. As the % of garbage lines depended on the number of chunks in a line, the splitting of extracted data was facilitated.
- The magic number of 2500 was chosen after some analysis described on `data_stats.md` that both other sources only ever have lines upto 1.7k chunks. So with a 50% margin we round up the 'big line length' to 2.5k.
- The minimum needed character density of 30% to make the line meaningful was derived from the long lines extracted during de-compression and was set as the standard to all other files before forming the training/testing corpus. 

---

## CC100 Tamil

The CC100 (Common Crawl 100) dataset is a massive, publicly available collection of monolingual text data used to pretrain natural language processing (NLP) models.

1. Extract the compressed file line by line and write to `tamil_cc100_extracted.txt`.
2. Using `lzma` to extract out lines from a compressed .xz file sequentially.
    - decode line in 'utf-8' format
    - strip links, urls, latin words
    - Split page text into smaller chunks to filter out unwated chars and only retain tamil alphabets+punctuation+digits
    - check if each line has atleast one tamil alphabet, only move to next step if it does

### Decision made
- The data in this source did not have deep nested formatting like xml to deal with, but it did have alot of dates and ill formatted numbers and obsecure translations like ki.mi for kilo metre among others. This was then used to refine the line stripping process to leave those kind of . inside and remove the recurring ones and ones that occur between non alphabets.
- As this is a webcrawl dataset, many of it's contents are duplicate. This was revealed in checks found in `data_stats.md`. As a result a whole deduplication step was added into the pipeline which reduced the useable unique lines by half of what it was before merging. As this was the biggest contributing data source with 68M lines any deduplication only makes sense after adding this to the mix.

---

## Clean Data

This is an intermediate step before the merginng of data and splitting it into train and test data. This was added after conducting data analysis on all the sources and the merged source to eliminate lines with only a single token. 

This single token in some cases ends up being an orphaned punctuation, a single word (which is often a header or a sub-heading translated form english and is sasme across pages such as `about us`) or even numbers (dates of publishing etc). 

These stand alone lines do not add any type of significance to the dataset but instead cause the pullution of it. Recurring headers and dates with no percieved value inflate the counts of a word occuring and hence add unnecessary and invaluable tokens to the vocabulary. 

A healthy threshold of 0.3 is used to decide if the line is legitemate or not. This is again due to the fact that some literature can only contain 2 tokens in a line.

All data sources are checked line by line and each of it is split into 2 files, with lines above and below the threshold. The 'above' files are the ones used in all downstream tasks including training the tokenizer and MLM fine-tuning. The 'below' files are to be inspected manually and for finding patterns in data which can make us take more informed decision and maybe change some decision rules as to what to include/ exclude in the following iterations.

---

## Corpus Merging and Train/Test split

1. Every 'above' file from the previous step are taken and merged into a single file, while merging: if any line does not have a single tamil alphabet that is discarded.
2. Iterate through lines of the merged file and use `hashlib` to hash each line. This is to find out if a line is unique. This step ensures all deplucate lines (predominantly from cc100) get filtered out and only one occurance of them stays. The deduped file is saved separately and the duplicate file is stored just for analysis like mentioned earlier.
3. On this deduped file, assign each line to either a test or a train set. We do this by selecting a train/test split then making a list of string for each and using `random.shuffle` to randomize assignment of lines to each set.
4. Write lines from the deduped file to `train.txt` and `test.txt` following the random assignment.

---

## Sanity Checks

Final step is to run a bunch of checks which test each file in the corpus and the relevant data sources. This was one of the steps used to build a robust pipeline. This is a final addendum to verify all of the scrits ran smoothly and the source is ready for downstream tasks. More information on this is provided in the `data_stats.md`.