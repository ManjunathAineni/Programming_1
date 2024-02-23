# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = Xtrain.max()
        answer["max_Xtest"] = Xtest.max()
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        

        answer = {}
        
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5, random_state=self.seed,shuffle=True)
        scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
        
        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        std_fit_time = scores["fit_time"].std()
        
        results_C = {
             "mean_accuracy":mean_accuracy,
             "std_accuracy":std_accuracy,
             "mean_fit_time":mean_fit_time,
             "std_fit_time":std_fit_time,            
        }
        
        answer["clf"] = clf  # the estimator (classifier instance)
        answer["cv"] = cv  # the cross validator instance
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        answer["scores"] = results_C
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {}
        
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, random_state=self.seed)
        scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
        results_D = {}

        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        std_fit_time = scores["fit_time"].std()
        
        results_D = {
             "mean_accuracy":mean_accuracy,
             "std_accuracy":std_accuracy,
             "mean_fit_time":mean_fit_time,
             "std_fit_time":std_fit_time,            
        }
        
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = results_D
        answer["explain_kfold_vs_shuffle_split"] = 'K-fold cross-validation partitions data into K subsets, using each as validation once. Shuffle-split randomly samples data for train/test sets'
        
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`

        answer = {}
        
        given_k = [2, 5, 8, 16]
        

        for k in given_k:
            
            clf = DecisionTreeClassifier(random_state=self.seed)
            
            cv = ShuffleSplit(n_splits=k, random_state=self.seed)
            
            scores_E = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
            
            results_E = {}
            
            for key, values in scores_E.items():
                if key == 'test_score':
                    results_E['mean_accuracy'] = values.mean()
                    results_E['std_accuracy'] = values.std()
                    
            answer[k] = {'scores': results_E, 'cv': cv, 'clf': clf}

        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """

        answer = {}
        
        res_part_D = self.partD(X,y)
        
        clf = RandomForestClassifier(random_state=self.seed)
        cv_ss = ShuffleSplit(n_splits=5,random_state=self.seed)
        ran_tree_ss = u.train_simple_classifier_with_cv(Xtrain=X,ytrain=y,clf=clf,cv=cv_ss)
        ran_key_ss ={}
        ran_key_ss['mean_fit_time'] = ran_tree_ss['fit_time'].mean()
        ran_key_ss['std_fit_time'] = ran_tree_ss['fit_time'].std()
        ran_key_ss['mean_accuracy'] = ran_tree_ss['test_score'].mean()
        ran_key_ss['std_accuracy'] = ran_tree_ss['test_score'].std()
        
        answer['clf_RF'] = clf
        answer['clf_DT'] = res_part_D['clf']
        answer['cv'] = cv_ss
        answer['scores_RF'] = ran_key_ss
        answer['scores_DT'] = res_part_D['scores']
        
        if res_part_D['scores']['mean_accuracy'] < ran_key_ss['mean_accuracy']:
            model_highest_accuracy = "Random Forest"
        else:
            model_highest_accuracy =  "Decision Tree"
            
        answer['model_highest_accuracy']= model_highest_accuracy
        
        partd_std = res_part_D['scores']['std_accuracy']
        var_partd = partd_std**2

        std = ran_key_ss['std_accuracy']
        var_std = std**2

        #std_accuracy = min(var_partd,var_std)
        
        if var_partd < var_std:
            model_lowest_variance = "Decision Tree"
        else:
            model_lowest_variance =  "Random Forest"
            
            
        
        answer['model_lowest_variance'] = model_lowest_variance
        
        partd_time = res_part_D['scores']['mean_fit_time']
        time = ran_key_ss['mean_fit_time']

        #model_fastest = min(partd_time,time)
        
        if partd_time < time:
            model_fastest = "Decision Tree"
        else:
            model_fastest =  "Random Forest"
        
        answer['model_fastest'] = model_fastest

        # Enter your code, construct the `answer` dictionary, and return it.

        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        answer = {}
        accuracies = {}
        
        def compute_accuracy(confusion_matrix):
            """
            Calculate accuracy from a confusion matrix.
            """
    
            TruePositive = confusion_matrix[1, 1]  # True Positives
            TrueNegative = confusion_matrix[0, 0]  # True Negatives
            total_samples = confusion_matrix.sum() 
    
            accuracy = (TruePositive+ TrueNegative) / total_samples
    
            return accuracy
    
           
        clf = RandomForestClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, random_state=self.seed)
        
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
        
        g_s = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, n_jobs=-1)
        g_s.fit(X, y)
        
        # Get the best estimator
        best_clf = g_s.best_estimator_
        
        # Fit the initial classifier on the entire training dataset
        clf.fit(X, y)
        
        # Make predictions using both classifiers
        y_train_pred_orig = clf.predict(X)
        y_train_pred_best = best_clf.predict(X)
        y_test_pred_orig = clf.predict(Xtest)
        y_test_pred_best = best_clf.predict(Xtest)
        
        # Compute confusion matrices
        confusion_matrix_train_orig = confusion_matrix(y, y_train_pred_orig)
        confusion_matrix_train_best = confusion_matrix(y, y_train_pred_best)
        confusion_matrix_test_orig = confusion_matrix(ytest, y_test_pred_orig)
        confusion_matrix_test_best = confusion_matrix(ytest, y_test_pred_best)
        
        # Calculate accuracy for each confusion matrix
        accuracies["accuracy_orig_full_training"] = compute_accuracy(confusion_matrix_train_orig)
        accuracies["accuracy_orig_full_testing"] = compute_accuracy(confusion_matrix_test_orig)
        accuracies["accuracy_best_full_training"] = compute_accuracy(confusion_matrix_train_best)
        accuracies["accuracy_best_full_testing"] = compute_accuracy(confusion_matrix_test_best) 
        
        
        answer = {
        "clf": clf,
        "default_parameters": clf.get_params(),
        "best_estimator": best_clf,
        "grid_search": g_s,
        "mean_accuracy_cv": g_s.best_score_,
        "confusion_matrix_train_orig": confusion_matrix_train_orig,
        "confusion_matrix_train_best": confusion_matrix_train_best,
        "confusion_matrix_test_orig": confusion_matrix_test_orig,
        "confusion_matrix_test_best": confusion_matrix_test_best,
        "accuracy_orig_full_training": accuracies["accuracy_orig_full_training"],
        "accuracy_best_full_training": accuracies["accuracy_best_full_training"],
        "accuracy_orig_full_testing": accuracies["accuracy_orig_full_testing"],
        "accuracy_best_full_testing": accuracies["accuracy_best_full_testing"]
        }
        
        
                # Enter your code, construct the `answer` dictionary, and return it.
        
        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        return answer
