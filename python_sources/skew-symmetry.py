---
title: 'Asymmetry and skew-symmetry in Scientific Migration'
date: '`r Sys.Date()`'
output:
  html_document:
    number_sections: true
    df_print: paged
    fig_caption: true
    toc: true
    #fig_width: 7
    #fig_height: 4.5
    theme: cosmo
    highlight: tango
    code_folding: hide
---

```{r setup, include=FALSE, echo=FALSE}
knitr::opts_chunk$set(
	echo = TRUE,
	message = FALSE,
	warning = FALSE
)
```
# Read data and libraries
```{r step1}
library(asymmetry)
library(RColorBrewer)
migration <- read.csv('../input/ORCID_migrations_2016_12_16_by_person.csv')
head(migration)
```
# Convert to migration matrix

We calculate the one step migration matrix from the country where the PHD is obtained to the country where they are working in 2016.
The rows of this matrix refer tot phd countries, whereas the columns of this matrix refer to the country in 2016. To conserve space a small example with three countries is shown below.
The entries read as follows: 1808 phd students stayed in Spain after the obtained their PHD, 4 moved to India, 85 moved to Germany, and so on. The analysis reported here uses the R package asymmetry.
```{r matrix}
mig2 <- migration[migration$has_migrated=='True',]
transitions <- as.matrix(table(mig2$phd_country,mig2$country_2016))
rows <- rownames(transitions)
cols <- colnames(transitions)
names <- rows[rows %in% cols==TRUE]
names <- names[2:length(names)]
square <- transitions[names,names]
square[c("ES","IN","DE"),c("ES","IN","DE")]
skew <- .5*(square-t(square))
# creates a color palette from red to blue
my_palette <- colorRampPalette(c("red", "white", "blue"))(n = 299)
col_breaks = c(seq(-4000,-.001,length=100),               # negative values are red
               seq(-.001,0.01,length=100),                # zeroes are white
               seq(0.01,4000,length=100))                 # positive values are blue
idx <- rowSums(abs(skew))>200
```
In this analysis having a large number of countries consumes a lot of space on a computerscreen, we reduce the number of countries by including those countries that have an inflow or outflow greater than 200.


# Skew-symmetry
The decomposition of an asymmetric matrix into a symmetric matrix and a skew-symmetric matrix is an elementary result from mathematics. The decomposition into a skew-symmetric and a symmetric component is written as:
$$ Q = S + A, $$
where $S$ is a symmetric matrix with averages $(q_{ij}+q_{ji})/2$, and $A$ is a skew-symmetric matrix with elements $(q_{ij}-q_{ji})/2$. A square matrix is skew-symmetric if the transpose can be obtained by multiplying the elements of the matrix by minus one, that is $A^T = -A$. A second property is $a_{ij}=-a_{ji}$, that is, if we interchange the subscripts the sign of the element changes. To summarize, by subtracting the average we obtain a skew-symmetric matrix, the elements of this matrix represent change in scientist population of a country. The skew-symmetric matrix is shown below.


```{r example}
q1 <- skewsymmetry(square[idx,idx])
q1$A[c("ES","IN","DE"),c("ES","IN","DE")]
```
The decomposition is additive, and because the two components $S$ and $A$ are orthogonal, the decomposition of the sum of squares of the two matrices is also additive.
```{r example2}
summary(q1)
```

# Display Heatmap
A heatmap displays the values in a datamatrix, and this heatmap displays the values of a skew-symmetric matrix, where the magnitude of the skew-symmetry is represented by color intensity, and the sign by the color red or blue. 
The option dominance is used in this analysis, and orders the rows and columns of the matrix in such a way that the values in the uppertriangle are positive and the values in the lower triangle are negative. The order is calculated from the row-sums of the signs obtained from the skew-symmetric matrix, and ignores the size of the migrations.
From the order of the countries we see that South-Afrika gains phd students from all other countries, and that net migrations from the Netherlands to other countries are higher than in the opposite direction.
```{r heat}
hmap(q1$A,col = my_palette)
```