relacounts <- read.csv(
  "~/Documents/projects/UMLS-Relation-extractor/rela_counts.txt",
  sep = '\t', na.strings = "NULL")

plot(relacounts$CNT[1:nrow(relacounts)-1], type = 'l')
plot(relacounts$CNT[800:1000], type = 'l')
# inflection point circa 975

relacounts.cumsum <- cumsum(relacounts$CNT[1:nrow(relacounts)-1])
plot(relacounts.cumsum, type = 'l')
# inflection point is circa 950
relacounts.cumsum[900]/relacounts.cumsum[1007]
# < 12% relations in the first 900 relation types

shaz = c(30846, 32607, 33209, 33857, 35222, 35244, 40137, 42457, 42601, 44305, 49182, 57378, 60915, 72775, 72775, 78102, 78779, 99079, 99079, 100135, 100172, 101060, 103824, 104108, 114878, 122914, 149167, 164697, 262612, 324512, 336048, 348990)
plot(shaz, type='l')
plot(cumsum(shaz), type='l')
length(shaz)
