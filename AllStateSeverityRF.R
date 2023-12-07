##Libraries
library(tidyverse)
library(tidymodels)
library(embed)
library(themis)
library(doParallel)

cl <- parallel::makePSOCKcluster(5)
doparallel::registerDoParallel(cl)

trainCsv <- read_csv("train.csv") %>%
  mutate_at(vars(cat1:cat116), as.factor)

testCsv <- read_csv("test.csv") %>%
  mutate_at(vars(cat1:cat116), as.factor)


#Create the recipe and bake it

rf_recipe <- recipe(loss ~ ., data=trainCsv) %>%
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .01)#%>%
#step_smote(all_outcomes(), neighbors=5)

prep <- prep(rf_recipe)
baked <- bake(prep, new_data = NULL)
baked



#Set up the model
my_mod <- rand_forest(mtry = 5,
                      min_n=20,
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

## Create a workflow with model & recipe
rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(my_mod)
  
  
  
  ## Set up grid of tuning values
  
  #CV Results 1,23
tuning_grid <- grid_regular(mtry(c(1,5)),
                            min_n(),
                              levels = 5)## L^2 total tuning possibilities

## Set up K-fold CV
folds <- vfold_cv(trainCsv, v = 3, repeats=1)

## Run the CV
CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
                 metrics=metric_set(mae)) #Or leave metrics NULL

## Find best tuning parameters
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="mae") %>%
  ggplot(data=., aes(x=mtry, y=min_n, color=factor(mtry))) +
  geom_line()

collect_metrics(CV_results)

CV_results
## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("mae")
bestTune

## Finalize the Workflow & fit it
final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

rf_predictions <- final_wf %>%
  predict(new_data = testCsv)

Sub1 <- rf_predictions %>% 
  bind_cols(testCsv) %>% 
  select(id,.pred_1) %>%
  rename(loss = .pred_1)


write_csv(Sub1, "RFSubmission.csv")

stopCluster(cl)
