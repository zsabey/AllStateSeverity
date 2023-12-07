##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(themis)
library(doParallel)

cl <- parallel::makePSOCKcluster(5)
doParallel::registerDoParallel(cl)

train <- vroom('train.csv')
test <- vroom('test.csv')

#Create the recipe and bake it

rf_recipe <- recipe(loss ~ ., data=train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))




#Set up the model
my_mod <- rand_forest(mtry = 5,
                      min_n=2,
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Create a workflow with model & recipe
rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(my_mod) %>%
  fit(data=train)
  
  
  
  ## Set up grid of tuning values
  
  #CV Results 1,23
# tuning_grid <- grid_regular(mtry(c(1,5)),
#                             min_n(),
#                               levels = 5)## L^2 total tuning possibilities
# 
# ## Set up K-fold CV
# folds <- vfold_cv(train, v = 3, repeats=1)
# 
# ## Run the CV
# CV_results <- rf_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#                  metrics=metric_set(mae)) #Or leave metrics NULL
# 
# ## Find best tuning parameters
# collect_metrics(CV_results) %>% # Gathers metrics into DF
#   filter(.metric=="mae") %>%
#   ggplot(data=., aes(x=mtry, y=min_n, color=factor(mtry))) +
#   geom_line()
# 
# collect_metrics(CV_results)
# 
# CV_results
# ## Find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best("mae")
# bestTune
# 
# ## Finalize the Workflow & fit it
# final_wf <- rf_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=train)

rf_predictions <- rf_wf %>%
  predict(new_data = test)

Sub1 <- rf_predictions %>% 
  bind_cols(test) %>% 
  select(id,.pred) %>%
  rename(loss = .pred)


write_csv(Sub1, "RFSubmission.csv")

stopCluster(cl)
