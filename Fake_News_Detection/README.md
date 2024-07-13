# Fake News Detection via Text Classification

## Task 1 – Explore Essential Information from Text Data and Preprocessing

### Most Common 100 Words

| Rank | word_counts_all         | word_counts_fake         | word_counts_true         |
|------|-------------------------|--------------------------|--------------------------|
| 1    | (said, 130050)          | (trump, 73744)           | (said, 99042)            |
| 2    | (trump, 128096)         | (said, 31008)            | (’, 70768)               |
| 3    | (’, 70768)              | (president, 26073)       | (trump, 54352)           |
| 4    | (u, 63450)              | (people, 26031)          | (“, 54140)               |
| 5    | (state, 58336)          | (one, 23682)             | (”, 53861)               |
| 6    | (would, 54945)          | (would, 23420)           | (u, 41166)               |
| 7    | (“, 54140)              | (u, 22284)               | (state, 36385)           |
| 8    | (”, 53861)              | (state, 21951)           | (would, 31525)           |
| 9    | (president, 53070)      | (clinton, 18595)         | (reuters, 28403)         |
| 10   | (people, 41354)         | (like, 18139)            | (president, 26997)       |
| 11   | (republican, 38106)     | (obama, 17760)           | (republican, 22109)      |
| 12   | (one, 36737)            | (time, 17692)            | (government, 19466)      |
| 13   | (year, 33507)           | (donald, 17101)          | (year, 18769)            |
| 14   | (also, 31174)           | (american, 16013)        | (house, 16934)           |
| 15   | (new, 30921)            | (republican, 15997)      | (new, 16786)             |
| 16   | (reuters, 28766)        | (say, 15440)             | (also, 15953)            |
| 17   | (government, 28519)     | (also, 15221)            | (united, 15574)          |
| 18   | (clinton, 28113)        | (year, 14738)            | (people, 15323)          |
| 19   | (house, 27646)          | (new, 14135)             | (party, 14990)           |
| 20   | (donald, 27554)         | (news, 14099)            | (official, 14580)        |
| 21   | (obama, 26955)          | (image, 13831)           | (told, 14244)            |
| 22   | (time, 26818)           | (even, 13659)            | (country, 14079)         |
| 23   | (say, 25385)            | (hillary, 13532)         | (election, 13959)        |
| 24   | (country, 24799)        | (white, 13114)           | (could, 13710)           |
| 25   | (could, 23899)          | (right, 12489)           | (one, 13055)             |
| 26   | (united, 23538)         | (get, 12207)             | (last, 12632)            |
| 27   | (told, 23344)           | (know, 11918)            | (washington, 12427)      |
| 28   | (election, 23172)       | (make, 11522)            | (two, 11624)             |
| 29   | (party, 22996)          | (via, 11164)             | (campaign, 11113)        |
| 30   | (american, 22878)       | (woman, 11159)           | (group, 11113)           |
| 31   | (like, 22777)           | (campaign, 11068)        | (former, 10601)          |
| 32   | (white, 22616)          | (medium, 11057)          | (leader, 10521)          |
| 33   | (campaign, 22181)       | (country, 10720)         | (week, 10485)            |
| 34   | (official, 20986)       | (house, 10712)           | (donald, 10453)          |
| 35   | (last, 20457)           | (america, 10585)         | (security, 10408)        |
| 36   | (right, 20389)          | (could, 10189)           | (court, 10357)           |
| 37   | (news, 20094)           | (first, 9986)            | (percent, 9948)          |
| 38   | (two, 19932)            | (want, 9807)             | (say, 9945)              |
| 39   | (group, 19042)          | (going, 9745)            | (north, 9872)            |
| 40   | (first, 18552)          | (think, 9727)            | (minister, 9542)         |
| 41   | (washington, 17901)     | (many, 9690)             | (clinton, 9518)          |
| 42   | (law, 17862)            | (way, 9351)              | (white, 9502)            |
| 43   | (make, 17678)           | (election, 9213)         | (law, 9297)              |
| 44   | (former, 17664)         | (day, 9172)              | (tax, 9245)              |
| 45   | (even, 17573)           | (told, 9100)             | (senate, 9221)           |
| 46   | (week, 16697)           | (government, 9053)       | (obama, 9195)            |
| 47   | (get, 16585)            | (thing, 8915)            | (time, 9126)             |
| 48   | (many, 16411)           | (made, 8662)             | (vote, 9010)             |
| 49   | (day, 16378)            | (law, 8565)              | (month, 8764)            |
| 50   | (hillary, 16273)        | (video, 8564)            | (china, 8593)            |
| 51   | (security, 16065)       | (back, 8560)             | (first, 8566)            |
| 52   | (vote, 16041)           | (police, 8537)           | (national, 8536)         |
| 53   | (court, 15857)          | (go, 8395)               | (statement, 8527)        |
| 54   | (national, 15728)       | (two, 8308)              | (administration, 8420)   |
| 55   | (want, 15579)           | (black, 8019)            | (since, 8332)            |
| 56   | (medium, 15505)         | (party, 8006)            | (tuesday, 8268)          |
| 57   | (may, 15438)            | (show, 7977)             | (democratic, 8240)       |
| 58   | (political, 15250)      | (united, 7964)           | (foreign, 8197)          |
| 59   | (made, 14875)           | (group, 7929)            | (including, 8119)        |
| 60   | (woman, 14858)          | (last, 7825)             | (military, 8052)         |
| 61   | (democrat, 14823)       | (take, 7784)             | (wednesday, 8012)        |
| 62   | (leader, 14756)         | (come, 7749)             | (presidential, 8012)     |
| 63   | (police, 14612)         | (see, 7706)              | (democrat, 7951)         |
| 64   | (million, 14451)        | (may, 7626)              | (right, 7900)            |
| 65   | (image, 14430)          | (political, 7545)        | (russia, 7853)           |
| 66   | (know, 14416)           | (fact, 7340)             | (may, 7812)              |
| 67   | (since, 14319)          | (national, 7192)         | (political, 7705)        |
| 68   | (percent, 14173)        | (report, 7174)           | (support, 7669)          |
| 69   | (bill, 14150)           | (need, 7146)             | (thursday, 7662)         |
| 70   | (going, 14106)          | (well, 7079)             | (bill, 7614)             |
| 71   | (support, 14061)        | (former, 7063)           | (million, 7559)          |
| 72   | (administration, 13984) | (vote, 7031)             | (policy, 7523)           |
| 73   | (think, 13878)          | (world, 6964)            | (plan, 7404)             |
| 74   | (take, 13822)           | (much, 6916)             | (friday, 7332)           |
| 75   | (way, 13789)            | (million, 6892)          | (korea, 7267)            |
| 76   | (back, 13737)           | (democrat, 6872)         | (day, 7206)              |
| 77   | (presidential, 13703)   | (life, 6700)             | (monday, 7099)           |
| 78   | (month, 13340)          | (story, 6650)            | (force, 7077)            |
| 79   | (statement, 13303)      | (bill, 6536)             | (office, 6953)           |
| 80   | (america, 13132)        | (public, 6509)           | (committee, 6886)        |
| 81   | (russia, 13118)         | (official, 6406)         | (american, 6865)         |
| 82   | (member, 13104)         | (support, 6392)          | (member, 6844)           |
| 83   | (democratic, 13006)     | (man, 6311)              | (deal, 6838)             |
| 84   | (tax, 12915)            | (attack, 6278)           | (many, 6721)             |
| 85   | (senate, 12721)         | (member, 6260)           | (agency, 6538)           |
| 86   | (policy, 12694)         | (week, 6212)             | (congress, 6499)         |
| 87   | (including, 12613)      | (according, 6205)        | (senator, 6486)          |
| 88   | (office, 12519)         | (never, 6180)            | (federal, 6447)          |
| 89   | (north, 12437)          | (another, 6171)          | (department, 6360)       |
| 90   | (according, 12346)      | (really, 6147)           | (issue, 6336)            |
| 91   | (attack, 12296)         | (family, 6138)           | (city, 6323)             |
| 92   | (report, 12115)         | (every, 6011)            | (company, 6229)          |
| 93   | (need, 12034)           | (since, 5987)            | (made, 6213)             |
| 94   | (department, 11989)     | (candidate, 5978)        | (make, 6156)             |
| 95   | (public, 11881)         | (work, 5873)             | (part, 6143)             |
| 96   | (go, 11788)             | (case, 5749)             | (according, 6141)        |
| 97   | (federal, 11757)        | (still, 5727)            | (comment, 6133)          |
| 98   | (world, 11746)          | (presidential, 5691)     | (police, 6075)           |
| 99   | (come, 11683)           | (child, 5689)            | (called, 6047)           |
| 100  | (via, 11673)            | (muslim, 5677)           | (take, 6038)             |

By having many more instances of the word "said" in the true news data set, it is evident that real news samples show a much higher incidence of direct quotes, also evidenced by the high count of single and double quotation marks. Fake news data seems to also employ the use of emphasis/absolute words such as "never," "even," or "really" at a higher rate than the real news data. Furthermore, these fake news texts also seem to use more words meant to incite a reaction such as "attack" or contain more demographic-based words such as "black" or "muslim."

As for the strongest feature set, it might be useful to focus on POS tagging with noun, verb, and adjective/adverb filters to highlight these differences. A way to look into context usage of high-frequency words might also be helpful.

## Task 2 – Build Machine Learning Model

### Model Performance:

![image](https://github.com/user-attachments/assets/43bdb3c5-c9a4-4786-97e6-ca5886358c2a)

Confusion Matrix for the top two models:

![image](https://github.com/user-attachments/assets/2c6fe5ab-0a41-45cf-b6c2-d760b73af3db)

The confusion matrices above show the number of true positive, true negative, false positive, and false negative predictions from each model. Both models appear to have a preference for false positives over false negatives, meaning they prefer to err on the side of caution with these predictions. Both models are performing well, with the vast majority of news articles being labeled correctly.

## Task 3 – Enhanced NLP Features

![image](https://github.com/user-attachments/assets/2a58d581-cfcb-4dd1-a79f-140767a5b963)

Below is a table outlining how POS tagging affected the performance of the models. Overall these models seem to mostly have improved, with the highest improvement in percentage coming from the Noun + Adjective filters with TF-IDF features. This generally indicates that paying close attention to the nouns and adjectives from the news data may provide important insight as to the authenticity of the texts, but given the mixed performance (especially with the Multinomial Naive Bayes model), more feature engineering strategies are necessary to better understand the model and its features, as well as to improve accuracy further.

| ML Model                    | Feature Filter            | Precision Change (%) | Recall Change (%) | Accuracy Change (%) |
|-----------------------------|---------------------------|----------------------|-------------------|---------------------|
| Logistic Regression         | TFIDF Noun + Adjective    | +0.27                | +0.26             | +0.27               |
| Support Vector Machine      | TF Noun + Adjective       | -0.02                | -0.02             | -0.03               |
| Gradient Boosting Machine   | TF Noun + Adjective + Verb| +0.1                 | +0.09             | +0.09               |
| Gradient Boosting Machine   | TFIDF Noun + Adjective    | +1.01                | +1.01             | +1.01               |
| Support Vector Machine      | TFIDF Noun + Verb         | +0.06                | +0.07             | +0.07               |
| Random Forest               | TF Noun + Adjective + Verb| +0.24                | +0.23             | +0.24               |
| Random Forest               | TFIDF Noun + Adjective    | +0.76                | +0.76             | +0.76               |
| Multinomial Naive Bayes     | TFIDF Noun + Verb         | +0.31                | +0.31             | +0.32               |
| Logistic Regression         | TF Noun + Adjective + Verb| -0.03                | -0.04             | -0.03               |
| Multinomial Naive Bayes     | TFIDF Noun + Adjective    | -0.19                | -0.18             | -0.18               |

