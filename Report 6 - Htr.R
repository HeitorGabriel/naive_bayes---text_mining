#######   Directory

# additional guide is in:
# https://www.r-bloggers.com/2021/03/text-mining-term-frequency-analysis-and-word-cloud-creation-using-the-tm-package/
setwd("/home/heitor/Área de Trabalho/R Projects/Análise Macro/Lab 6")

#######   Packages Used

library(tidyverse)   # standard
library(tidymodels)  # standard
library(tm)          # text mining
library(SnowballC)   # to stemming words
library(wordcloud2)  # make words visualizations, the input have to be a data.frame
library(wordcloud)
library(naivebayes)  # naive bayes procedures
library(gmodels)     # to tabulate results

#######   Import and Treatment 

#dd <- read_csv("sms_spam.csv", col_types = cols(type = col_factor(levels = c("ham","spam"))))
dd <- read_csv("sms_spam.csv") %>% as_tibble()

dd$type <- as.factor(dd$type)

dd %>% glimpse() %>% summary()

# Making a Corpus text from a simple vector:
dd_corpus1 <- VCorpus(VectorSource(dd$text))

# Clean the corpus:
clean_corpus <- function(corpus_to_use){
  corpus_to_use %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace) %>%
    tm_map(content_transformer(function(x) iconv(x, to='UTF-8', sub='byte'))) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords("en")) %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removeWords, c("etc","ie", "eg", stopwords("english"))) %>% 
    tm_map(stemDocument)
}
dd_corpus2 <- clean_corpus(dd_corpus1)
remove(dd_corpus1)

#dd_corpus2 %>% inspect()
# to see how is:
lapply(dd_corpus2[1:3], as.character)

#  Make data with frequency of words

find_freq_terms_fun <- function(corpus_in){
   dd_dtm        <- DocumentTermMatrix(corpus_in)
   freq_terms    <- findFreqTerms(dd_dtm)[1:max(dd_dtm$ncol)]
   terms_grouped <- dd_dtm[,freq_terms] %>%
     as.matrix() %>%
     colSums() %>%
     data.frame(Term=freq_terms, Frequency = .) %>%
     arrange(desc(Frequency)) %>%
     mutate(prop_term_to_total_terms=Frequency/nrow(.))
   return(data.frame(terms_grouped))
}
freq_terms_crp <- data.frame(find_freq_terms_fun(dd_corpus2))
View(freq_terms_crp)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
wc1 <- wordcloud2(freq_terms_crp[,1:2] %>%
                    filter(freq_terms_crp$Frequency>35),
                  shape="circle",
                  color="random-light",
                  backgroundColor = "black")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#   Cut words that appears in less than 5 texts
#freq_terms_crp2 <- freq_terms_crp %>% filter(freq_terms_crp$Frequency >= 5)
# wue dont delete these words anymore cause we'll work with DTM data, not freq. data
# do this to others analysis with freq data.

# Separate Train and Numeric

# The following two methods dont works with DTM
#sep <- initial_split(dd_dtm, prop = 0.80)
#dd_train <- training(sep)
#dd_test  <- testing(sep)
#       and
#train <- dd_dtm %>% sample_frac(.,0.8)
#sid   <- as.numeric(rownames(train)) # because rownames() returns character
#test  <- dt[-sid,] %>% select(-Churn)

dtm <- DocumentTermMatrix(dd_corpus2)

dtm_train <- dtm[1:4169, ]
dtm_test  <- dtm[4170:5559, ]

# store the types of the DTM's docs and verify if train and test have the same proportion
train_type <- dd[1:4169, ]$type
test_type  <- dd[4170:5559, ]$type
prop.table(table(train_type))
prop.table(table(train_type))

# we have to decrease the data, dropping the infrequent words
freq_words <- findFreqTerms(dtm_train, 5)
# create DTMs with only the frequent terms
dtm_train2 <- dtm_train [ , freq_words]
dtm_test2  <- dtm_test  [ , freq_words]

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() convert_counts() to columns of train/test data
dd_train <- apply(dtm_train2, MARGIN = 2, convert_counts)
dd_test  <- apply(dtm_test2,  MARGIN = 2, convert_counts)

#######   Applying the model

nb1 <- naive_bayes(x = dd_train,
                   y = train_type,
                   laplace= 1,
                   usepoisson = T,
                   usekernel = T)
#nb11 <- naive_bayes(x = dd_train,
#                   y = train_type,
#                   laplace= 1,
#                   usepoisson = T,
#                   usekernel = F)
#
#nb2 <- naive_bayes(x = dd_train,
#                   y = train_type,
#                   laplace= 1,
#                   usepoisson = F,
#                   usekernel = T)
#nb22 <- naive_bayes(x = dd_train,
#                   y = train_type,
#                   laplace= 1,
#                   usepoisson = F,
#                   usekernel = F)

tables(nb1, c('call', 'pay', 'love', 'free', 'now'))

# use the model in test set
tst1 <- predict(nb1, dd_test,
                type= 'class')
#tst11 <- predict(nb11, dd_test,
#                type= 'class')
#tst2 <- predict(nb2, dd_test,
#                type= 'class')
#tst22 <- predict(nb22, dd_test,
#                type= 'class')

table(tst1, test_type)
#table(tst11, test_type)
#table(tst2, test_type)
#table(tst22, test_type)

CrossTable(tst1, test_type,
           prop.chisq = FALSE,
           prop.t = T,
           prop.r = F,
           prop.c = F,
           dnn = c('predicted', 'actual'))
#CrossTable(tst11, test_type,
#           prop.chisq = FALSE,
#           prop.t = T,
#           prop.r = F,
#           prop.c = F,
#           dnn = c('predicted', 'actual'))
#CrossTable(tst2, test_type,
#           prop.chisq = FALSE,
#           prop.t = T,
#           prop.r = F,
#           prop.c = F,
#           dnn = c('predicted', 'actual'))
#CrossTable(tst22, test_type,
#           prop.chisq = FALSE,
#           prop.t = T,
#           prop.r = F,
#           prop.c = F,
#           dnn = c('predicted', 'actual'))














