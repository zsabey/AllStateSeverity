##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(themis)
library(vroom)

cl <- parallel::makePSOCKcluster(5)
doParallel::registerDoParallel(cl)

train <- vroom('train.csv')
test <- vroom('test.csv')
#skimr::skim(train)
### EDA #### 
#hist(train$loss)
### Some variables are highly correlated to each other 
#plot_correlation(train, type = 'continuous')
train$loss <- log(train$loss)
### Initial_Split ####
boost <- boost_tree(mode = 'regression', 
                    learn_rate = tune(),
                    loss_reduction = tune(),
                    tree_depth = tune(),
                    min_n = tune()
) %>%
  set_engine('xgboost', objective = 'reg:absoluteerror')
boost_recipe <- recipe(loss ~ ., train) %>% 
  update_role(id, new_role = 'ID') %>%
  step_scale(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = .6) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  prep()

boost_workflow <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost) 


## Set up grid of tuning values

#CV Results 1,23
tuning_grid <- grid_regular(tree_depth(),
                            loss_reduction(),
                            trees(),
                            learn_rate(),
                            levels = 5)## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(train, v = 3, repeats=1)

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
  fit(data=train)

boost_predictions <- final_wf %>%
  predict(new_data = test)


Sub1 <- boost_predictions %>% 
  bind_cols(test) %>% 
  select(id,.pred) %>%
  rename(loss = .pred)


write_csv(Sub1, "boostSubmission.csv")
stopCluster(cl)

