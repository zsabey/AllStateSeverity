##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(themis)

##431498


cl <- parallel::makePSOCKcluster(5)
doParallel::registerDoParallel(cl)


trainCsv <- read_csv("train.csv") %>%
  mutate_at(vars(cat1:cat116), as.factor)

testCsv <- read_csv("test.csv") %>%
  mutate_at(vars(cat1:cat116), as.factor)


#Create the recipe and bake it

boost_recipe <- recipe(loss ~ ., data=trainCsv) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))%>%
  step_zv() %>%
  step_normalize(all_predictors()) #%>%
  #step_pca(all_predictors(), threshold = .01)
#step_smote(all_outcomes(), neighbors=5)




#Set up the model
boost_model <- boost_tree(tree_depth=tune(),
                         trees=tune(),
                         learn_rate=tune()) %>%
  set_engine("xgboost") %>% #or "xgboost" but lightgbm is faster
  set_mode("regression")

## Create a workflow with model & recipe
boost_workflow <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model) 



## Set up grid of tuning values

#CV Results 1,23
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 3)## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- boost_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(mae)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("mae")
bestTune

## Finalize the Workflow & fit it
final_wf <- boost_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

boost_predictions <- final_wf %>%
  predict(new_data = testCsv)


Sub1 <- boost_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred) %>%
  rename(loss = .pred)


write_csv(Sub1, "boostSubmission.csv")
stopCluster(cl)

