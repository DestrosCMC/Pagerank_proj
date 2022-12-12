# Pagerank Project

In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.

## Update 12/12/2022  
Updated project by implementing word2vec.  

Collaborated w/ Hanane for part 1 of word2vec.

**Due date:** Sunday, 18 September at midnight

**Computation:**
This project has low computational requirements.
You are not required to complete it on the lambda server (although you are welcome to if you'd like).

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the `P` matrix,
this is also the value of `nnz(P)`.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of `P`, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the `FIXME` annotation.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the `P` matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the `P` matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their `P` matrix for the web,
they use a similar (but much more complicated) process to modify the `P` matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the `\bar\bar P` matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the `P` graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank_solution.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank_solution.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.
If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _url_to_index function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).

1. Run the following commands, and paste their output into the code blocks below.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=0.3775096535682678
    DEBUG:root:i=1 residual=0.3134882152080536
    DEBUG:root:i=2 residual=0.2756592035293579
    DEBUG:root:i=3 residual=0.21698090434074402
    DEBUG:root:i=4 residual=0.18984204530715942
    DEBUG:root:i=5 residual=0.15531454980373383
    DEBUG:root:i=6 residual=0.13266263902187347
    DEBUG:root:i=7 residual=0.11062280833721161
    DEBUG:root:i=8 residual=0.0935136154294014
    DEBUG:root:i=9 residual=0.07847193628549576
    DEBUG:root:i=10 residual=0.06611069291830063
    DEBUG:root:i=11 residual=0.05558090656995773
    DEBUG:root:i=12 residual=0.04677915573120117
    DEBUG:root:i=13 residual=0.039349012076854706
    DEBUG:root:i=14 residual=0.03310869261622429
    DEBUG:root:i=15 residual=0.027853948995471
    DEBUG:root:i=16 residual=0.023434894159436226
    DEBUG:root:i=17 residual=0.019716130569577217
    DEBUG:root:i=18 residual=0.016587838530540466
    DEBUG:root:i=19 residual=0.013955758884549141
    DEBUG:root:i=20 residual=0.011741633526980877
    DEBUG:root:i=21 residual=0.009878222830593586
    DEBUG:root:i=22 residual=0.008310933597385883
    DEBUG:root:i=23 residual=0.006992380600422621
    DEBUG:root:i=24 residual=0.00588278379291296
    DEBUG:root:i=25 residual=0.00494928564876318
    DEBUG:root:i=26 residual=0.004164101090282202
    DEBUG:root:i=27 residual=0.003503275103867054
    DEBUG:root:i=28 residual=0.002947412896901369
    DEBUG:root:i=29 residual=0.0024798011872917414
    DEBUG:root:i=30 residual=0.002086422871798277
    DEBUG:root:i=31 residual=0.001755191246047616
    DEBUG:root:i=32 residual=0.0014767624670639634
    DEBUG:root:i=33 residual=0.0012423915322870016
    DEBUG:root:i=34 residual=0.0010452595306560397
    DEBUG:root:i=35 residual=0.0008794325985945761
    DEBUG:root:i=36 residual=0.0007398635498248041
    DEBUG:root:i=37 residual=0.0006225021206773818
    DEBUG:root:i=38 residual=0.0005237152217887342
    DEBUG:root:i=39 residual=0.00044049162534065545
    DEBUG:root:i=40 residual=0.0003707956930156797
    DEBUG:root:i=41 residual=0.0003118595341220498
    DEBUG:root:i=42 residual=0.00026235461700707674
    DEBUG:root:i=43 residual=0.0002207769575761631
    DEBUG:root:i=44 residual=0.00018579662719275802
    DEBUG:root:i=45 residual=0.0001563065015943721
    DEBUG:root:i=46 residual=0.0001315098925260827
    DEBUG:root:i=47 residual=0.00011065506259910762
    DEBUG:root:i=48 residual=9.283208783017471e-05
    DEBUG:root:i=49 residual=7.835238648112863e-05
    DEBUG:root:i=50 residual=6.579793989658356e-05
    DEBUG:root:i=51 residual=5.543866427615285e-05
    DEBUG:root:i=52 residual=4.6715060307178646e-05
    DEBUG:root:i=53 residual=3.909929364454001e-05
    DEBUG:root:i=54 residual=3.294386260677129e-05
    DEBUG:root:i=55 residual=2.7852673156303354e-05
    DEBUG:root:i=56 residual=2.3248065190273337e-05
    DEBUG:root:i=57 residual=1.966203490155749e-05
    DEBUG:root:i=58 residual=1.6471843991894275e-05
    DEBUG:root:i=59 residual=1.4014856787980534e-05
    DEBUG:root:i=60 residual=1.1800590982602444e-05
    DEBUG:root:i=61 residual=9.742259862832725e-06
    DEBUG:root:i=62 residual=8.302875357912853e-06
    DEBUG:root:i=63 residual=7.063716111588292e-06
    DEBUG:root:i=64 residual=5.845966825290816e-06
    DEBUG:root:i=65 residual=4.962869752489496e-06
    DEBUG:root:i=66 residual=4.206880475976504e-06
    DEBUG:root:i=67 residual=3.499711510812631e-06
    DEBUG:root:i=68 residual=2.992129338963423e-06
    DEBUG:root:i=69 residual=2.5033950805664062e-06
    DEBUG:root:i=70 residual=2.214214191553765e-06
    DEBUG:root:i=71 residual=1.955177822310361e-06
    DEBUG:root:i=72 residual=1.3902072169003077e-06
    DEBUG:root:i=73 residual=1.244581540049694e-06
    DEBUG:root:i=74 residual=9.97376446321141e-07
    INFO:root:rank=0 ranking=2.1634e+00 url=4
    INFO:root:rank=1 ranking=1.6664e+00 url=6
    INFO:root:rank=2 ranking=1.2402e+00 url=5
    INFO:root:rank=3 ranking=4.5712e-01 url=2
    INFO:root:rank=4 ranking=3.5620e-01 url=3
    INFO:root:rank=5 ranking=3.2078e-01 url=1

   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
    INFO:root:rank=0 ranking=1.5478e-13 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=1 ranking=1.3655e-13 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=2 ranking=1.0640e-13 url=www.lawfareblog.com/dc-circuits-thoroughly-convincing-decision-al-nashiri
    INFO:root:rank=3 ranking=8.8141e-14 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=4 ranking=8.5691e-14 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=5 ranking=7.9507e-14 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
    INFO:root:rank=6 ranking=7.7273e-14 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
    INFO:root:rank=7 ranking=7.6703e-14 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
    INFO:root:rank=8 ranking=7.6006e-14 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
    INFO:root:rank=9 ranking=7.3839e-14 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns

   

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'  
    INFO:root:rank=0 ranking=2.7966e-09 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=1 ranking=3.6279e-10 url=www.lawfareblog.com/did-donald-trump-jr-admit-violating-computer-fraud-and-abuse-act
    INFO:root:rank=2 ranking=1.4583e-10 url=www.lawfareblog.com/documents-saifullah-paracha-v-donald-j-trump
    INFO:root:rank=3 ranking=8.9874e-11 url=www.lawfareblog.com/cta9-decides-al-nashiri-v-macdonald
    INFO:root:rank=4 ranking=7.2813e-11 url=www.lawfareblog.com/donald-trump-danger-our-national-security
    INFO:root:rank=5 ranking=6.5934e-11 url=www.lawfareblog.com/strategic-underpinning-and-limits-republican-due-process-defense-donald-trump
    INFO:root:rank=6 ranking=6.5588e-11 url=www.lawfareblog.com/burden-donald-trump
    INFO:root:rank=7 ranking=6.1771e-11 url=www.lawfareblog.com/nashiri-v-macdonald-dismissed
    INFO:root:rank=8 ranking=6.1474e-11 url=www.lawfareblog.com/does-trump-want-lose-eo-battle-court-or-donald-mcgahn-simply-ineffectual-or-worse
    INFO:root:rank=9 ranking=6.1321e-11 url=www.lawfareblog.com/donald-trumps-pardon-power-and-state-exception

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'  
    INFO:root:rank=0 ranking=4.5624e-07 url=www.lawfareblog.com/update-military-commissions-continued-health-issues-recusal-motion-and-new-cell-al-iraqi
    INFO:root:rank=1 ranking=3.9028e-07 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
    INFO:root:rank=2 ranking=3.1269e-07 url=www.lawfareblog.com/france-makes-play-try-foreign-fighters-iraq
    INFO:root:rank=3 ranking=1.9591e-07 url=www.lawfareblog.com/its-not-only-iraq-and-syria
    INFO:root:rank=4 ranking=1.6319e-07 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=5 ranking=1.6229e-07 url=www.lawfareblog.com/document-sens-kaine-and-young-introduce-bill-revoke-iraq-force-authorizations
    INFO:root:rank=6 ranking=1.2054e-07 url=www.lawfareblog.com/assessing-aclu-habeas-petition-behalf-unnamed-us-citizen-held-enemy-combatant-iraq
    INFO:root:rank=7 ranking=1.1921e-07 url=www.lawfareblog.com/primer-can-trump-administration-transfer-american-citizen-enemy-combatant-iraqi-custody
    INFO:root:rank=8 ranking=5.4169e-08 url=www.lawfareblog.com/2002-iraq-aumf-almost-certainly-authorizes-president-use-force-today-iraq-and-might-authorize-use
    INFO:root:rank=9 ranking=4.2728e-08 url=www.lawfareblog.com/last-week-military-commissions-medical-accommodations-and-conspiracy-liability-al-iraqi
    
   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
    INFO:root:rank=0 ranking=8.4156e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=1 ranking=8.4156e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=2 ranking=8.4156e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=3 ranking=8.4156e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=4 ranking=8.4156e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=5 ranking=8.4156e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=6 ranking=8.4156e+00 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=7 ranking=8.4156e+00 url=www.lawfareblog.com/support-lawfare
    INFO:root:rank=8 ranking=8.4156e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 ranking=8.4156e+00 url=www.lawfareblog.com/our-comments-policy


   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
    INFO:root:rank=0 ranking=4.6091e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 ranking=2.9867e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 ranking=2.9669e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 ranking=2.0173e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 ranking=1.8769e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 ranking=1.8762e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 ranking=1.8693e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 ranking=1.7655e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 ranking=1.6807e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 ranking=9.8346e-01 url=www.lawfareblog.com/events

   
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=20.519933700561523
    DEBUG:root:i=1 residual=6.110218524932861
    DEBUG:root:i=2 residual=1.9214706420898438
    DEBUG:root:i=3 residual=0.5871003866195679
    DEBUG:root:i=4 residual=0.17524230480194092
    DEBUG:root:i=5 residual=0.05138478800654411
    DEBUG:root:i=6 residual=0.014827270992100239
    DEBUG:root:i=7 residual=0.00416865898296237
    DEBUG:root:i=8 residual=0.001077273627743125
    DEBUG:root:i=9 residual=0.0001930922007886693
    DEBUG:root:i=10 residual=8.493969653500244e-05
    DEBUG:root:i=11 residual=0.00013263747678138316
    DEBUG:root:i=12 residual=0.000128910323837772
    DEBUG:root:i=13 residual=0.00010905414092121646
    DEBUG:root:i=14 residual=9.583001519786194e-05
    DEBUG:root:i=15 residual=7.931196159915999e-05
    DEBUG:root:i=16 residual=6.609127012779936e-05
    DEBUG:root:i=17 residual=5.617739225272089e-05
    DEBUG:root:i=18 residual=4.626503505278379e-05
    DEBUG:root:i=19 residual=3.96539326175116e-05
    DEBUG:root:i=20 residual=3.634970198618248e-05
    DEBUG:root:i=21 residual=3.3044216252164915e-05
    DEBUG:root:i=22 residual=2.6437381166033447e-05
    DEBUG:root:i=23 residual=1.982917638088111e-05
    DEBUG:root:i=24 residual=1.6522233636351302e-05
    DEBUG:root:i=25 residual=1.652348873903975e-05
    DEBUG:root:i=26 residual=1.3217977539170533e-05
    DEBUG:root:i=27 residual=9.91389879345661e-06
    DEBUG:root:i=28 residual=9.912546374835074e-06
    DEBUG:root:i=29 residual=9.912531822919846e-06
    DEBUG:root:i=30 residual=3.3123531011369778e-06
    DEBUG:root:i=31 residual=6.608392141060904e-06
    DEBUG:root:i=32 residual=6.608376224903623e-06
    DEBUG:root:i=33 residual=3.3059432098525576e-06
    DEBUG:root:i=34 residual=3.3041985716408817e-06
    DEBUG:root:i=35 residual=6.144799158391834e-08
    INFO:root:rank=0 ranking=8.4156e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=1 ranking=8.4156e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=2 ranking=8.4156e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=3 ranking=8.4156e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=4 ranking=8.4156e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=5 ranking=8.4156e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=6 ranking=8.4156e+00 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=7 ranking=8.4156e+00 url=www.lawfareblog.com/support-lawfare
    INFO:root:rank=8 ranking=8.4156e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 ranking=8.4156e+00 url=www.lawfareblog.com/our-comments-policy

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=24.14374542236328
    DEBUG:root:i=1 residual=8.458536148071289
    DEBUG:root:i=2 residual=3.128673791885376
    ...
    DEBUG:root:i=998 residual=0.0003139341133646667
    DEBUG:root:i=999 residual=0.0003139339096378535
    INFO:root:rank=0 ranking=1.0624e+01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=1 ranking=1.0624e+01 url=www.lawfareblog.com/masthead
    INFO:root:rank=2 ranking=1.0624e+01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=3 ranking=1.0624e+01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=4 ranking=1.0624e+01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=5 ranking=1.0624e+01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=6 ranking=1.0624e+01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=7 ranking=1.0624e+01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=8 ranking=1.0624e+01 url=www.lawfareblog.com/topics
    INFO:root:rank=9 ranking=1.0624e+01 url=www.lawfareblog.com/support-lawfare


   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
    
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=5.117237567901611
    DEBUG:root:i=1 residual=2.9804813861846924
    DEBUG:root:i=2 residual=2.4491045475006104
    DEBUG:root:i=3 residual=1.698635220527649
    DEBUG:root:i=4 residual=1.1000990867614746
    DEBUG:root:i=5 residual=0.7297568321228027
    DEBUG:root:i=6 residual=0.514761209487915
    DEBUG:root:i=7 residual=0.3795803487300873
    DEBUG:root:i=8 residual=0.2850838005542755
    DEBUG:root:i=9 residual=0.2157047986984253
    DEBUG:root:i=10 residual=0.16483207046985626
    DEBUG:root:i=11 residual=0.12841403484344482
    DEBUG:root:i=12 residual=0.10304608941078186
    DEBUG:root:i=13 residual=0.08559361845254898
    DEBUG:root:i=14 residual=0.07334396988153458
    DEBUG:root:i=15 residual=0.06423044949769974
    DEBUG:root:i=16 residual=0.05689764395356178
    DEBUG:root:i=17 residual=0.05057094991207123
    DEBUG:root:i=18 residual=0.044861793518066406
    DEBUG:root:i=19 residual=0.03961014002561569
    DEBUG:root:i=20 residual=0.03476380184292793
    DEBUG:root:i=21 residual=0.030314506962895393
    DEBUG:root:i=22 residual=0.02626786194741726
    DEBUG:root:i=23 residual=0.022626224905252457
    DEBUG:root:i=24 residual=0.01938277669250965
    DEBUG:root:i=25 residual=0.016521476209163666
    DEBUG:root:i=26 residual=0.014020247384905815
    DEBUG:root:i=27 residual=0.011850166134536266
    DEBUG:root:i=28 residual=0.00998082384467125
    DEBUG:root:i=29 residual=0.008380168117582798
    DEBUG:root:i=30 residual=0.007017120253294706
    DEBUG:root:i=31 residual=0.005861947312951088
    DEBUG:root:i=32 residual=0.004886546637862921
    DEBUG:root:i=33 residual=0.00406653480604291
    DEBUG:root:i=34 residual=0.0033789058215916157
    DEBUG:root:i=35 residual=0.002803673967719078
    DEBUG:root:i=36 residual=0.0023237913846969604
    DEBUG:root:i=37 residual=0.0019244913710281253
    DEBUG:root:i=38 residual=0.0015928337816148996
    DEBUG:root:i=39 residual=0.0013172643957659602
    DEBUG:root:i=40 residual=0.0010889500845223665
    DEBUG:root:i=41 residual=0.0008999287383630872
    DEBUG:root:i=42 residual=0.0007435237057507038
    DEBUG:root:i=43 residual=0.0006142214988358319
    DEBUG:root:i=44 residual=0.0005074203363619745
    DEBUG:root:i=45 residual=0.000419005926232785
    DEBUG:root:i=46 residual=0.00034628110006451607
    DEBUG:root:i=47 residual=0.00028601399390026927
    DEBUG:root:i=48 residual=0.00023641210282221437
    DEBUG:root:i=49 residual=0.00019538190099410713
    DEBUG:root:i=50 residual=0.0001612970809219405
    DEBUG:root:i=51 residual=0.00013338716235011816
    DEBUG:root:i=52 residual=0.00011045379505958408
    DEBUG:root:i=53 residual=9.132847480941564e-05
    DEBUG:root:i=54 residual=7.552596798632294e-05
    DEBUG:root:i=55 residual=6.261348607949913e-05
    DEBUG:root:i=56 residual=5.184937617741525e-05
    DEBUG:root:i=57 residual=4.296215047361329e-05
    DEBUG:root:i=58 residual=3.5611814382718876e-05
    DEBUG:root:i=59 residual=2.9574126529041678e-05
    DEBUG:root:i=60 residual=2.4380973627557978e-05
    DEBUG:root:i=61 residual=2.02734736376442e-05
    DEBUG:root:i=62 residual=1.6909214537008666e-05
    DEBUG:root:i=63 residual=1.3909829249314498e-05
    DEBUG:root:i=64 residual=1.160780266218353e-05
    DEBUG:root:i=65 residual=9.660861906013452e-06
    DEBUG:root:i=66 residual=8.071858246694319e-06
    DEBUG:root:i=67 residual=6.802133611927275e-06
    DEBUG:root:i=68 residual=5.655035693052923e-06
    DEBUG:root:i=69 residual=4.621172593033407e-06
    DEBUG:root:i=70 residual=3.877404651575489e-06
    DEBUG:root:i=71 residual=3.2103971534525044e-06
    DEBUG:root:i=72 residual=2.699147444218397e-06
    DEBUG:root:i=73 residual=2.4067735466815066e-06
    DEBUG:root:i=74 residual=1.8685715303945472e-06
    DEBUG:root:i=75 residual=1.5895365095275338e-06
    DEBUG:root:i=76 residual=1.3939329619461205e-06
    DEBUG:root:i=77 residual=1.143126951319573e-06
    DEBUG:root:i=78 residual=9.235662901119213e-07
    INFO:root:rank=0 ranking=4.6091e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 ranking=2.9867e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 ranking=2.9669e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 ranking=2.0173e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 ranking=1.8769e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 ranking=1.8762e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 ranking=1.8693e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 ranking=1.7655e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 ranking=1.6807e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 ranking=9.8346e-01 url=www.lawfareblog.com/events

    
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
   
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=6.019961357116699
    DEBUG:root:i=1 residual=4.125228404998779
    DEBUG:root:i=2 residual=3.987910032272339
    DEBUG:root:i=3 residual=3.2539920806884766
    DEBUG:root:i=998 residual=3.0866136512486264e-05
    DEBUG:root:i=999 residual=2.813212813634891e-05
    INFO:root:rank=0 ranking=5.2385e+01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 ranking=5.2385e+01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 ranking=7.9438e+00 url=www.lawfareblog.com/cost-using-zero-days
    INFO:root:rank=3 ranking=2.3700e+00 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
    INFO:root:rank=4 ranking=1.5529e+00 url=www.lawfareblog.com/events
    INFO:root:rank=5 ranking=1.1867e+00 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
    INFO:root:rank=6 ranking=1.1867e+00 url=www.lawfareblog.com/water-wars-drill-maybe-drill
    INFO:root:rank=7 ranking=1.1867e+00 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
    INFO:root:rank=8 ranking=1.1867e+00 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
    INFO:root:rank=9 ranking=1.1867e+00 url=www.lawfareblog.com/water-wars-us-china-divide-shangri-la


   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
    INFO:root:rank=0 ranking=7.1729e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 ranking=7.1727e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 ranking=1.4754e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 ranking=1.1785e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=4 ranking=1.1785e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=5 ranking=8.7547e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=6 ranking=8.5943e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=7 ranking=8.1309e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=8 ranking=8.1278e-02 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=9 ranking=8.1278e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism

   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'  
    INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
    INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america

   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You made trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   Each part is worth 2 points, for 12 points overall.
