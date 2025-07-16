# Load libraries
library(tm)
library(topicmodels)
library(tidytext)
library(ggplot2)
library(dplyr)
library(stringr)
library(tidyr)
library(textstem)
library(patchwork)
remotes::install_github("nikita-moor/ldatuning") #3
library(ldatuning)



# Set working directory to where your "sport_1" folder is located
setwd("D:/UKM/Master/2.Unstructed_Data/Project 1/")  
# Load corpus from the folder
corpus <- VCorpus(DirSource("news", encoding = "UTF-8"))

# Preprocessing with lemmatization
#custom_words <- c("game", "team", "match", "player")  # your chosen words to remove
#toSpace <- content_transformer(function(x, pattern) {return (gsub(pattern, " ", x))})
#corpus <- tm_map(corpus, toSpace, "-")
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
#corpus <- tm_map(corpus, removeWords, custom_words)
corpus <- tm_map(corpus, stripWhitespace)
#lemmatize_corpus <- content_transformer(function(x) lemmatize_strings(x))
#corpus <- tm_map(corpus, lemmatize_corpus)
#corpus <- tm_map(corpus, stemDocument)

# Create Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)  # Remove rarely used words

# Find optimal number of topics using coherence and perplexity-based methods
result <- FindTopicsNumber(
  dtm,
  topics = seq(2, 10, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",  # because your model uses Gibbs sampling (LDA)
  control = list(seed = 1234),
  mc.cores = 2L,  # adjust if needed
  verbose = TRUE
)

# Plot the results
FindTopicsNumber_plot(result)

# ====================================================
# 1. Topic Modelling (k = 3)
# ====================================================
lda_k3 <- LDA(dtm, k = 3, control = list(seed = 1234))

# Extract topic-word probabilities
beta_k3 <- tidy(lda_k3, matrix = "beta")

# Top 8 terms per topic
top_terms_k3 <- beta_k3 %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

print(top_terms_k3, n=100)

# Plot
top_terms_k3_plot <-top_terms_k3 %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top 10 Terms per Topic (k = 3)", x = NULL, y = "Probability")


# Step 1: Reshape topic-term matrix (beta_k3) into wide format
beta_spread_k3 <- beta_k3 %>%
  mutate(topic = paste0("topic", topic)) %>%
  pivot_wider(names_from = topic, values_from = beta)

# Step 2: Filter for terms that appear meaningfully in both topics
# Let's compare topic1 and topic3
beta_diff_k3 <- beta_spread_k3 %>%
  filter(topic1 > 0.005 | topic3 > 0.005) %>%  # keep common or prominent terms
  mutate(log_ratio = log2(topic3 / topic1))

print(beta_diff_k3,n=100)

# Step 3: Plot log ratio to show difference
beta_diff_k3 %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_col(show.legend = FALSE, fill = "steelblue") +
  coord_flip() +
  labs(title = "Difference between Topic 3 and Topic 1 (log2 ratio)", x = NULL, y = "Log2(Topic3 / Topic1)")


# ====================================================
# 2. Per-document Topic Distribution (Gamma)
# ====================================================
gamma_k3 <- tidy(lda_k3, matrix = "gamma")

# View the topic mix for some documents
head(gamma_k3)

# Plot
gamma_k3_plot<-ggplot(gamma_k3, aes(document, gamma, fill = factor(topic))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Topic Proportions per Document (k = 3)", x = "Document", y = "Probability") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

# ====================================================
# 5. Sentiment Analysis by Topic using Bing (k = 3)
# ====================================================

# Step 1: Join topic-term matrix with Bing sentiment lexicon
bing_topic_sentiment_k3 <- beta_k3 %>%
  inner_join(get_sentiments("bing"), by = c(term = "word"))

# Step 2: Aggregate sentiment scores by topic
bing_sentiment_by_topic_k3 <- bing_topic_sentiment_k3 %>%
  group_by(topic, sentiment) %>%
  summarise(total_beta = sum(beta), .groups = "drop")

# Step 3: Plot Bing sentiment by topic
bing_topic_plot_k3 <- ggplot(bing_sentiment_by_topic_k3, aes(x = factor(topic), y = total_beta, fill = sentiment)) +
  geom_col(position = "dodge") +
  labs(title = "Bing Sentiment by Topic (k = 3)", x = "Topic", y = "Sum of Word Probabilities")

# Display
bing_topic_plot_k3

# ====================================================
# 6. Sentiment Analysis by Topic using NRC (k = 3)
# ====================================================

# Step 1: Join topic-term matrix with NRC sentiment lexicon
nrc_topic_sentiment_k3 <- beta_k3 %>%
  inner_join(get_sentiments("nrc"), by = c(term = "word"))

# Step 2: Aggregate NRC sentiment scores by topic
nrc_sentiment_by_topic_k3 <- nrc_topic_sentiment_k3 %>%
  group_by(topic, sentiment) %>%
  summarise(total_beta = sum(beta), .groups = "drop")

print(nrc_sentiment_by_topic_k3, n=100)

# Step 3: Plot NRC sentiment by topic
nrc_topic_plot_k3 <- ggplot(nrc_sentiment_by_topic_k3, aes(x = reorder(sentiment, total_beta), y = total_beta, fill = factor(topic))) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "NRC Sentiment by Topic (k = 3)", x = "Sentiment", y = "Sum of Word Probabilities")

# Display
nrc_topic_plot_k3

# ====================================================
# 5. TF-IDF Analysis by Topic
# ====================================================
# Convert DTM to tidy format
library(tidytext)
tidy_dtm <- tidy(dtm)

# Calculate TF-IDF
tfidf <- tidy_dtm %>%
  bind_tf_idf(term, document, count)

# Step 1: Get per-document topic assignments from gamma
topic_doc_k3 <- tidy(lda_k3, matrix = "gamma") %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>%  # Get most dominant topic for each document
  ungroup()

# Step 2: Merge with TF-IDF data
tfidf_topic <- tfidf %>%
  inner_join(topic_doc_k3, by = "document")

# Step 3: Get top TF-IDF terms per topic
top_tfidf_by_topic <- tfidf_topic %>%
  group_by(topic) %>%
  top_n(10, tf_idf) %>%
  ungroup()

print(top_tfidf_by_topic, n=100)

# Step 4: Plot TF-IDF by topic
tf_k3<-ggplot(top_tfidf_by_topic, aes(x = reorder_within(term, tf_idf, topic), y = tf_idf, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top TF-IDF Terms by Topic (k = 3)", x = "Term", y = "TF-IDF Score")

## Compare with k=6--------------------------------------------------------------------------------------------------------------------
# ====================================================
# 1. Topic Modelling (k = 6)
# ====================================================
lda_k6 <- LDA(dtm, k = 6, control = list(seed = 1234))
beta_k6 <- tidy(lda_k6, matrix = "beta")

# Top 8 terms per topic
top_terms_k6 <- beta_k6 %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

print(top_terms_k6, n=100)

# Plot top terms
top_terms_k6_plot<-top_terms_k6 %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top 10 Terms per Topic (k = 6)", x = NULL, y = "Probability")

# Step 1: Reshape topic-term matrix for k = 6
beta_spread_k6 <- beta_k6 %>%
  mutate(topic = paste0("topic", topic)) %>%
  pivot_wider(names_from = topic, values_from = beta)

# Step 2: Filter and compute log ratio for Topic 3 and Topic 1
beta_diff_k6 <- beta_spread_k6 %>%
  filter(topic1 > 0.005 | topic5 > 0.005) %>%  # adjust threshold if needed
  mutate(log_ratio = log2(topic5 / topic1))

print(beta_diff_k6, n=100)

# Step 3: Plot log ratio
beta_diff_k6 %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_col(show.legend = FALSE, fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Difference between Topic 5 and Topic 1 (log2 ratio) [k = 6]",
    x = NULL,
    y = "Log2(Topic5 / Topic1)"
  )

# ====================================================
# 2. Per-document Topic Distribution (Gamma)
# ====================================================
gamma_k6 <- tidy(lda_k6, matrix = "gamma")

gamma_k6_plot<-ggplot(gamma_k6, aes(document, gamma, fill = factor(topic))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Topic Proportions per Document (k = 6)", x = "Document", y = "Probability") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

# ====================================================
# 5. Sentiment Analysis by Topic using Bing (k = 6)
# ====================================================

bing_topic_sentiment_k6 <- beta_k6 %>%
  inner_join(get_sentiments("bing"), by = c(term = "word"))

bing_sentiment_by_topic_k6 <- bing_topic_sentiment_k6 %>%
  group_by(topic, sentiment) %>%
  summarise(total_beta = sum(beta), .groups = "drop")

bing_topic_plot_k6 <- ggplot(bing_sentiment_by_topic_k6, aes(x = factor(topic), y = total_beta, fill = sentiment)) +
  geom_col(position = "dodge") +
  labs(title = "Bing Sentiment by Topic (k = 6)", x = "Topic", y = "Sum of Word Probabilities")


# ====================================================
# 6. Sentiment Analysis by Topic using NRC (k = 6)
# ====================================================

nrc_topic_sentiment_k6 <- beta_k6 %>%
  inner_join(get_sentiments("nrc"), by = c(term = "word"))

nrc_sentiment_by_topic_k6 <- nrc_topic_sentiment_k6 %>%
  group_by(topic, sentiment) %>%
  summarise(total_beta = sum(beta), .groups = "drop")

print(nrc_sentiment_by_topic_k6, n=100)

nrc_topic_plot_k6 <- ggplot(nrc_sentiment_by_topic_k6, aes(x = reorder(sentiment, total_beta), y = total_beta, fill = factor(topic))) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "NRC Sentiment by Topic (k = 6)", x = "Sentiment", y = "Sum of Word Probabilities")
nrc_topic_plot_k6

# ====================================================
# 5. TF-IDF Analysis by Topic (k = 6)
# ====================================================

# Step 1: Get per-document topic assignments from gamma (k = 6)
topic_doc_k6 <- tidy(lda_k6, matrix = "gamma") %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>%  # Get most dominant topic for each document
  ungroup()

# Step 2: Merge with TF-IDF data
tfidf_topic_k6 <- tfidf %>%
  inner_join(topic_doc_k6, by = "document")

# Step 3: Get top TF-IDF terms per topic
top_tfidf_by_topic_k6 <- tfidf_topic_k6 %>%
  group_by(topic) %>%
  top_n(10, tf_idf) %>%
  ungroup()

# Step 4: Plot TF-IDF by topic (k = 6)
tf_k6<-ggplot(top_tfidf_by_topic_k6, aes(x = reorder_within(term, tf_idf, topic), y = tf_idf, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "Top TF-IDF Terms by Topic (k = 6)", x = "Term", y = "TF-IDF Score")

print(top_tfidf_by_topic_k6, n=100)



