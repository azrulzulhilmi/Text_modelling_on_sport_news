# 📰 Malaysia Sports News Text Analysis

<p align="center">
  <img src="pics/sport.png" alt="Malaysia Sports" width="20%">
</p>

[![R](https://img.shields.io/badge/R-4.3.1-blue.svg)](https://www.r-project.org/)
[![RStudio](https://img.shields.io/badge/IDE-RStudio-blue)](https://www.rstudio.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tidyverse](https://img.shields.io/badge/Library-tidyverse-orange)](https://www.tidyverse.org/)
[![Topic Modeling](https://img.shields.io/badge/Topic_Modeling-LDA-informational)](https://cran.r-project.org/web/packages/topicmodels/index.html)
[![SentimentR](https://img.shields.io/badge/Sentiment-lexicons-green)](https://cran.r-project.org/web/packages/syuzhet/index.html)
[![tm Package](https://img.shields.io/badge/TextMining-tm-informational)](https://cran.r-project.org/web/packages/tm/index.html)
[![ggplot2](https://img.shields.io/badge/Visualization-ggplot2-blueviolet)](https://ggplot2.tidyverse.org/)

---

## 📌 Overview

This project applies **topic modeling**, **sentiment analysis**, and **TF-IDF term extraction** to a corpus of 120 Malaysia sports news articles written in English, sourced from *New Straits Times (NST)* and *The Star (Malaysia)*. The goal is to uncover key topics, emotional tones, and term patterns to better understand how Malaysian sports are represented in the media.

---

## 📘 Objectives

- Discover dominant themes using **LDA Topic Modeling**
- Evaluate topic coherence for different topic counts (k = 3 vs k = 6)
- Perform sentiment analysis using **Bing** and **NRC** lexicons
- Extract **TF-IDF** terms to highlight unique keywords
- Visualize how documents are distributed across topics using **Gamma values**

---

## 🧪 Methodology

1. Collected 120 English sport articles (May-June,2025)
2. Preprocessing: lowercase conversion, stopword removal, punctuation cleaning, and lemmatization
3. Constructed Document-Term Matrix (DTM)
4. Applied **Latent Dirichlet Allocation (LDA)** using Gibbs Sampling
5. Evaluated topic coherence with `ldatuning` package
6. Visualized per-topic term probabilities (beta) and document proportions (gamma)
7. Performed **Bing** and **NRC** sentiment analysis
8. Applied **TF-IDF** to find unique document-level terms

---

## 🔍 LDA Tuning Results

![LDA Tuning Plot](pics/ldatuning.png)

**Discussion**:  
Multiple topic coherence metrics were used to determine the optimal number of topics. The Deveaud2014 and Griffiths2004 scores indicated that **k = 6** provides better topic separation and interpretability compared to lower k values.

---

## 📌 Top 10 Terms per Topic

### k = 3

![Top 10 Terms (k=3)](pics/top10_terms_k_3.png)

**Discussion**:  
With only 3 topics, broader themes are captured. For example, Topic 1 combines international sports, Topic 2 focuses on team-based games, and Topic 3 captures performance or seasonal insights. However, topic boundaries are less distinct.

---

### k = 6

![Top 10 Terms (k=6)](pics/top10_terms_k_6.png)

**Discussion**:  
The k = 6 model shows **clearer topic separation**. For instance, Topic 1 is about **badminton**, Topic 3 about **cycling**, Topic 4 about **football**, and Topic 5 captures **international hockey**. Each topic focuses on specific sports or issues like injuries, match performance, or global competitions.

---

## 📉 Beta Log-Ratio Comparison Between Topics

### 🔹 k = 3: Topic 3 vs Topic 1  
![Beta Log-Ratio k=3 (T3 vs T1)](pics/beta_topic3&1_k_3.png)

**Discussion**:  
This plot visualizes the **log2 ratio of per-topic word probabilities** between Topic 3 and Topic 1 in the *k = 3 model*.  
- Words on the right (positive values) are **more probable in Topic 3** (e.g. `cycle`, `race`, `sea`), indicating focus on **cycling or endurance sports**.
- Words on the left (negative values) are **more probable in Topic 1** (e.g. `bam`, `pair`, `tournament`), indicating **badminton**.
- Terms near 0 are **shared** across both topics, such as `national` or `win`.

---

### 🔹 k = 6: Topic 5 vs Topic 1  
![Beta Log-Ratio k=6 (T5 vs T1)](pics/beta_topic5&1_k_6.png)

**Discussion**:  
In the k = 6 model, this plot compares Topic 5 (likely **hockey or team sports**) against Topic 1 (**badminton**).  
- Positive values (right) highlight words strongly linked to Topic 5, such as `goal`, `match`, `minutes`, and `Pakistan`.
- Negative values (left) show badminton-centric terms for Topic 1 like `bam`, `badminton`, and `open`.
- This analysis demonstrates clearer separation between sports in the k = 6 model.

---

## ❤️ Bing Sentiment Analysis

![Bing Sentiment (k=3 and k=6)](pics/bing_lexicon.png)

**Discussion**:  
All topics show higher positive sentiment than negative. Topics like double pair wins in badminton and local tournaments carry optimistic tone. Topics discussing injuries (e.g., Topic 2 in k=6) had slightly more balanced or neutral tones.

---

## 😃 NRC Emotion Analysis

![NRC Sentiment (k=3 and k=6)](pics/nrc_lexicons.png)

**Discussion**:  
Positive emotions like **trust**, **anticipation**, and **joy** dominate. Negative emotions like **anger** or **fear** are minimal and mostly linked to injury-related topics. Topic 3 in both models consistently shows high positive emotion, indicating uplifting sports narratives.

---

## 🏷️ TF-IDF Top Terms by Topic

![TF-IDF Terms](pics/tf_idf.png)

**Discussion**:  
TF-IDF analysis highlights **document-specific keywords**. Names like “Razif”, “Sidek” and “Iskandar” appear in athlete-focused articles. Terms like “Selangor” or “Homebois” indicate team-related coverage. This method complements topic modeling by surfacing unique language patterns.

---

## 📊 Topic Proportions per Document (Gamma)

![Gamma Plot](pics/gamma_matrix.png)

**Discussion**:  
Gamma values show most documents align strongly with a **single dominant topic** (values close to 1). This suggests the LDA model produces **coherent, non-overlapping topics**, especially in the k = 6 model.

---

## ✅ Conclusion

Both topic models offer valuable insights, but the **k = 6 model is more interpretable**, providing **better separation between sports themes** such as badminton, cycling, football and injuries. Gamma values further illustrated strong topic alignment per document. **TF-IDF terms** helped highlight distinct terms within each topic. Sentiment analysis confirmed a generally positive tone across Malaysian sports news, supporting national pride and motivation.

---

## 👨‍🎓 Author 

**Author:** Azrul Zulhilmi Ahmad Rosli  

---
