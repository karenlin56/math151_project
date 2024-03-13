########################### Read in the data ####################################
library(R.matlab)
library(ggplot2)
library(rpart)
library(ipred) 
library(modelr)
library(caret)
library(purrr)
library("Metrics")
library(randomForest)
library(boot)

set.seed(101)
traffic.flow <- readMat("C:\\Users\\karen\\Documents\\College\\SJSU\\Math 151 Spr 23\\Project\\Data\\Traffic Flow\\traffic_dataset.mat")

################################################################################
##                     LOOK AT DATA IN ITS ORIGINAL FORM                      ##
################################################################################

###Structure of data
typeof(traffic.flow)
class(traffic.flow)
class(traffic.flow$tra.X.tr)
class(traffic.flow$tra.Y.tr)
class(traffic.flow$tra.X.te)
class(traffic.flow$tra.Y.te)
str(traffic.flow)
head(traffic.flow)

## Training predictors
## Not possible to turn into a dataframe
traffic.flow$tra.X.tr[1]  ## 1st matrix
traffic.flow$tra.X.tr[[1]]
traffic.flow$tra.X.tr[[1]][[1]]
traffic.flow$tra.X.tr[[1]][[1]][100] ## 1st entry of traffic.flow$tra.X.tr (1st matrix), 1st row, 1st col. 
traffic.flow$tra.X.tr[[1]][[1]][5]
## How to transform this list into a dataframe? first 36 entries (rows) are the 
## values for the 1st feature for the 36 locations



## Training response variable
head(traffic.flow$tra.Y.tr) ## Just prints out the indices
traffic.flow$tra.Y.tr[1] ## look at first entry
traffic.flow$tra.Y.tr[1, 2]
class(traffic.flow$tra.Y.tr) ## Array, matrix, list? 
dim(traffic.flow$tra.Y.tr) ## dimensions (36 rows x 1261 columns) each row is a location, 
                           ## each col. is a time interval


## Testing predictors
traffic.flow$tra.X.te[[1]][[1]][2] ## 1st entry of test set, first row, 2nd column


## Testing response variable
traffic.flow$tra.Y.te[1, 1]


## Adjacency matrix that stores values of connectedness of roads
## Feature reduction? If some roads are connected....I don't know how
## to use that info to build a model
traffic.flow$tra.adj.mat

################################################################################
##                          Data cleaning                                     ##
################################################################################



#################  Make train and test data frames  ############################



### Make the dataframe consisting of only the time interval and first set of 
### historical values of the first 36 observational units (rows). Other columns 
### are added later. Time intervals are numbered in increasing order starting from 
### 1-(last time interval).
interval <- c() ## List that stores first 36 initial values of the "interval" column
for (i in 1:36) 
{
    interval <- append(interval, 1) ## First 36 observational units belong to the 
                                    ## first time interval so first 36 values 
                                    ## under the "interval" column are 1.
                                    ## However, the values don't really matter
                                    ## since I overwrite them a couple lines down. 
                                    ## These values act like initial placeholders.
}
interval

train.df <- data.frame(time.interval = interval, hist1 = traffic.flow$tra.X.tr[[1]][[1]][1:36])
train.df




## Add all columns from data in original form to dataframe I made
for (i in 1: length(traffic.flow$tra.X.tr)) ## Iterate through all of training
{                                           ## data 
  ##if (i != 1)
  ##{
      interval <- c() 
      for (k in 1:36) ## 36 rows of each entry of training data (each entry is 
                      ##                                        like a matrix)
      {
        interval <- append(interval, i)
      }
      train.df[(i*36-35):(i*36), 1] = interval ## Put the correct time interval 
                                               ## value under the "interval"
                                               ## column of the dataframe for the 
                                               ## 36 rows of each entry in the 
                                               ## original training set from 
                                               ## the UC Irvine ML repository
  ##} 
  ## Columns (48) of each ith interval
  for (j in 1:48)
  {
    
      col.j <- traffic.flow$tra.X.tr[[i]][[1]][(j*36-35):(36*j)]
      train.df[(i*36-35):(i*36),j+1] = col.j ## Insert column entries corresponding 
                                             ## to the appropriate rows belonging 
                                             ## to ith time interval
  }
}
head(train.df) ## It takes a while since large loop with lots of things to do in 
               ## in each iteration


##### Rename columns of dataframe and make columns that correct datatypes #####
## 1) 10 historical values 1-10
## 2) weekday (7 binary) 11-17 (V12 or V17)
## 3) hour of day (24 binary columns) 18-41 (V42, V45, V46, or V48; V45 or V48)
## 4) road direction (4 binary columns north, west, east, south?) 42-45 (V46 or V47?)
## 5) number of lanes (discrete) 46, column 40?
## 6) name of road (2 highways) 47 (V46 or V47?)
## It is unclear which column represents which variable.

### Rename columns (at least the columns which are obvious) ###
names(train.df)[names(train.df) == "V41"] <- "lanes"
names(train.df)[names(train.df) == "V3"] <- "hist2"
names(train.df)[names(train.df) == "V4"] <- "hist3"
names(train.df)[names(train.df) == "V5"] <- "hist4"
names(train.df)[names(train.df) == "V6"] <- "hist5"
names(train.df)[names(train.df) == "V7"] <- "hist6"
names(train.df)[names(train.df) == "V8"] <- "hist7"
names(train.df)[names(train.df) == "V9"] <- "hist8"
names(train.df)[names(train.df) == "V10"] <- "hist9"
names(train.df)[names(train.df) == "V11"] <- "hist10"





################################################

## Take a look at the head again 
head(train.df)
## Add a location column (There are 36)
train.df$location <- c(1,1) ## Put all 1's initially
for (i in 1:length(traffic.flow$tra.X.tr))
{
  for (j in 1:36) 
  {
    train.df[i*36-(36-j), "location"] = j
  }      
}
## 
head(train.df)






##################### Check for NA and NaN values ##############################
which(is.na(train.df)==TRUE) ## None. Great!

nan.values <- c() ## vector to store number of nan values of each column
for (i in 1:length(colnames(train.df)))
{
    nan.values <- append(nan.values, length(which(is.nan(train.df[,i])==TRUE)))
}
print(nan.values) ## Look at which columns have nan values. None. Great!






################ Add response variable to training dataframe  ##################
## Iterate through each column of response of training set from original data
## since columns from that matrix thing represent the time intervals

train.df$traffic.flow <- c(0,0)
## Go by row in dataframe I made and add value under response column
for (i in 1:nrow(train.df))##nrow(train.df2)
{
    interval <- ceiling(i/36) ##interval index
    
    if (i%%36 != 0)
    {
        location <- i%%36 ##location index
    }
    else
    {
        location <- 36
    }
    tf <- traffic.flow$tra.Y.tr[location, interval]
    train.df[i, "traffic.flow"] = tf
}
train.df[1:36,"traffic.flow"] ## Check to see if that worked



##############  Make sure columns are of the correct data type #################
head(train.df)
str(train.df)
train.df$location <- factor(train.df$location)
train.df$lanes <- factor(train.df$lanes) ## Should this be categorical? 

## Make V12-V49 categorical variables
for (i in 12:49)
{
    if (i != 41)
    {
        train.df[,i] <- factor(train.df[,i])
    }
}




################## Make testing set into a dataframe ###########################
interval.test <- c()
for (i in 1:36) ## Column for first time interval
{
  interval.test <- append(interval.test, 1)         
}
interval.test
test.df <- data.frame(time.interval = interval.test, hist1 = traffic.flow$tra.X.te[[1]][[1]][1:36])
test.df

## 1st column (training, testing)
for (i in 1: length(traffic.flow$tra.X.te))
{
  ##if (i != 1)
  ##{
  interval <- c() 
  for (k in 1:36)
  {
    interval <- append(interval, i)
  }
  test.df[(i*36-35):(i*36), 1] = interval
  ##}
  for (j in 1:48)
  {
    col.j <- traffic.flow$tra.X.te[[i]][[1]][(j*36-35):(36*j)]
    test.df[(i*36-35):(i*36),j+1] = col.j
  }
  
  
}
test.df[1:36,]



### Rename columns (at least the columns which are obvious) ###
names(test.df)[names(test.df) == "V41"] <- "lanes"
names(test.df)[names(test.df) == "V3"] <- "hist2"
names(test.df)[names(test.df) == "V4"] <- "hist3"
names(test.df)[names(test.df) == "V5"] <- "hist4"
names(test.df)[names(test.df) == "V6"] <- "hist5"
names(test.df)[names(test.df) == "V7"] <- "hist6"
names(test.df)[names(test.df) == "V8"] <- "hist7"
names(test.df)[names(test.df) == "V9"] <- "hist8"
names(test.df)[names(test.df) == "V10"] <- "hist9"
names(test.df)[names(test.df) == "V11"] <- "hist10"



## Add a location column (There are 36)
test.df$location <- c(1,1) ## Put all 1's initially
for (i in 1:length(traffic.flow$tra.X.te))
{
  for (j in 1:36) 
  {
    test.df[i*36-(36-j), "location"] = j
  }      
}
## 
head(test.df)



######## Make sure columns of test.df are of the correct data type 
str(test.df)
test.df$location <- factor(test.df$location)
test.df$lanes <- factor(test.df$lanes) ## Should this be categorical? 

## Make V12-V49 categorical variables
for (i in 12:49)
{
  if (i != 41)
  {
    test.df[,i] <- factor(test.df[,i])
  }
}

str(test.df)

######## Add response variable to testing  dataframe 
## Iterate through each column of response of training set from original data
## since columns from that matrix thing represent the time intervals
traffic.flow$tra.Y.te[1:36, 1]
test.df$traffic.flow <- c(0,0)
## Go by row in dataframe I made and add value under response column
for (i in 1:nrow(test.df))##nrow(train.df2)
{
  interval <- ceiling(i/36) ##interval index
  
  if (i%%36 != 0)
  {
    location <- i%%36 ##location index
  }
  else
  {
    location <- 36
  }
  tf <- traffic.flow$tra.Y.te[location, interval]
  test.df[i, "traffic.flow"] = tf
}
test.df[1:36,"traffic.flow"] ## Check to see if that worked

################################################################################
##                       EXPLORATORY DATA ANALYSIS                            ##
################################################################################
## Line graph (colored by location)

line.graph <- ggplot(train.df, aes(x=time.interval,y=traffic.flow))
line.graph <- line.graph + geom_point(size=0.01, aes(color=location))
line.graph ## Not very easy to look at so let's just use a subset of the data

## Subset of train.df2 (first 500 intervals and first 5 locations)
##Locations 1-5
subset_indices <- c(1,2,3,4,5) ## Vector storing indices of rows of subset
for (j in 1:5)
{
    ## Each row belonging to location i is 36 rows away from the next one
    for (i in 1:499)
    {
        subset_indices <- append(subset_indices, j+36*i)
    
    }   
}

train.df3 <- train.df[subset_indices, ]

line.graph2 <- ggplot(train.df3, aes(x=time.interval,y=traffic.flow))
line.graph2 <- line.graph2 + geom_point(size=3, aes(color=location))
line.graph2 <- line.graph2 + coord_cartesian(xlim=c(1,500))
line.graph2 ## Lots of overlap and hard to see so I will have one separate graph
            ## for each of the 5 locations


## Histogram (colored by location) (y-axis is traffic.flow)
histogrm <- ggplot(train.df3, aes(x=traffic.flow))
histogrm <- histogrm + geom_histogram(binwidth=0.1, aes(fill=location))
histogrm ## A little hard to distinguish which location contributes how much to 
         ## to traffic flow. Maybe change color palette? 


head(train.df)



1261/(1261+840)
1-1261/(1261+840)
################################################################################
##                           MODEL BUILDING                                   ##
################################################################################


####### Cross-validation #########
cv <- crossv_kfold(train.df, k=10)
cv
##################################


##############################   Tree   ########################################
## Tree without any pruning, complexity parameter (alpha) = 0
my.tree <- rpart(traffic.flow~., method="anova", data=train.df, control=list(cp=0))
my.tree
summary(my.tree)
plot(my.tree)
##text(my.tree) 



## Tree with pruning
my.tree2 <- rpart(traffic.flow~., method="anova", data=train.df)
plotcp(my.tree2) ## plot complexity parameters
plot(my.tree2)
text(my.tree2)
my.tree2$cptable


## Tree test MSE
tree2.predictions <- predict(my.tree2, test.df)
test.mse.tree <- mean((tree2.predictions-test.df$traffic.flow)^2)
test.mse.tree

### Cross-validation
cv.model.1 <- map(cv$train, ~rpart(traffic.flow~., method="anova", data=train.df, control=list(cp=0.01) ))
cv.model.1
predictions.cv.7 <- predict(cv.model.1[7], test.df)
predictions.cv.7 <- unlist(predictions.cv.7)
predictions.cv.7-test.df$traffic.flow
sum((predictions.cv.7-test.df$traffic.flow)^2)
mean(sum((predictions.cv.7-test.df$traffic.flow)^2)) ## Same as the one above. Change, 

tree.cv.mses <- c()
for (i in 1:length(cv.model.1)) 
{
  predictions.i <- predict(cv.model.1[i], test.df) 
  predictions.i <- unlist(predictions.i) ## For some reason is a list, and we need to be numeric?
  mse <- mean((predictions.i-test.df$traffic.flow)^2)
  tree.cv.mses <- append(tree.cv.mses, mse)
}
tree.cv.mse <- mean(tree.cv.mses)
tree.cv.mses
tree.cv.mse

##cv.model.2 <- map(cv$train, ~rpart(traffic.flow~., method="anova", data=., control=list(cp=0.01)))
##cv.model.2


#mse(test.df$traffic.flow, tree2.predictions)
############################  Bagging ##########################################
#my.bag <- bagging(formula=traffic.flow~., data=train.df, nbagg=100, coob=TRUE, control=list(cp=0))
my.bag2 <- bagging(formula=traffic.flow~., data=train.df, nbagg=100, coob=TRUE)
my.bag2
#my.bag2 ## Takes so long
predictions.bag <- predict(my.bag2, test.df)
## Test MSE
mse.bag <- mean((test.df$traffic.flow-predictions.bag)^2)
mse.bag
0.0633 ## out of the bag estimate of root mean squared error
0.0633^2 ## out of the bag estimate of mean squared error
## Cross-validation MSE CAREFUL! TAKES MORE THAN AN HOUR
cv.bag <- map(cv$train, ~bagging(formula=traffic.flow~., data=train.df, nbagg=100, coob=TRUE))
bag.cv.mses <- c()
for (i in 1:length(cv.bag)) 
{
  predictions.i <- predict(cv.bag[i], test.df) 
  predictions.i <- unlist(predictions.i)
  mse <- mean((predictions.i-test.df$traffic.flow)^2)
  bag.cv.mses <- append(bag.cv.mses, mse)
}
mean(bag.cv.mses)


#########################   Random forest ######################################
rf <- randomForest(traffic.flow~., mtry=7, data=train.df)

### Test MSE
rf.predictions <- predict(rf, test.df) ## predictions produced by random forest model on test set
##f.test.mse <- mean(rf.predictions-test.df$traffic.flow)^2)
cv.rf <- map(cv$train, ~randomForest(traffic.flow~., mtry=7, data=train.df))
rf.cv.mses <- c()
for (i in 1: length(cv.rf))
{
  predictions.i <- predict(cv.rf[i], test.df) 
  predictions.i <- unlist(predictions.i)
  mse <- (sum(predictions.i-test.df$traffic.flow)^2)/length(test.df$traffic.flow)
  rf.cv.mses <- append(rf.cv.mses, mse)  
}


#########################  Multiple Linear Regresion ###########################
lm.traffic.flow <- lm(traffic.flow~., data=train.df)
summary(lm.traffic.flow)[2]
colnames(train.df)

lm.traffic.flow.2 <- glm(traffic.flow~., data = train.df)


str(train.df)
train.df$location <- factor(train.df$location)


######## Check assumptions #########
## Check residuals are normally distributed
hist(lm.traffic.flow$residuals) 
## Check residuals are normally distributed
qqnorm(lm.traffic.flow$residuals) ##qqplot
qqline(lm.traffic.flow$residuals)
## Check for constant variance for residuals
plot(fitted(lm.traffic.flow, residuals(lm.traffic.flow)))
residuals.plot <- ggplot(data=train.df, aes(x=residuals(lm.traffic.flow), y=fitted(lm.traffic.flow)))
residuals.plot <- residuals.plot + geom_point()
residuals.plot



### Test MSE ###
linear.predictions <- predict(lm.traffic.flow, test.df)
mean((linear.predictions-test.df$traffic.flow)^2)

### Cross-validation ###
cv.lr <- rep(0, 10)
for (i in 1:10) 
{
  glm.fit <- glm(traffic.flow~., data = train.df)
  cv.lr[i] <- cv.glm(test.df , glm.fit, K=10)$delta [1]
}
cv.lr
mean(cv.lr)



