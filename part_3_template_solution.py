import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
import new_utils as nu
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,make_scorer, f1_score,accuracy_score, recall_score,precision_score,top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score,ShuffleSplit

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")
        ct={}
        for i,j in enumerate(counts):
            ct[uniq[i]]=j

        return {
            "class_counts": ct,  # Replace with actual class counts
            "num_classes": uniq,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}
        
        answer = {}
        clf=LogisticRegression(random_state=self.seed,max_iter=300)
        cv=ShuffleSplit(n_splits=5,random_state=self.seed)

        cv_scores = {k: [] for k in range(1, 6)}

        test_scores = {k: None for k in range(1, 6)}

        def top_k_accuracy_scorer(k):
            return make_scorer(top_k_accuracy_score, needs_proba=True, k=k)
 
        for k in range(1, 6):
            cv_score = cross_val_score(clf, Xtrain, ytrain, cv=cv, scoring=top_k_accuracy_scorer(k))
            cv_scores[k] = cv_score.mean()

        clf.fit(Xtrain, ytrain)
        for k in range(1, 6):
            y_test_proba = clf.predict_proba(Xtest)
            test_scores[k] = top_k_accuracy_score(ytest, y_test_proba, k=k)
   
        answer['clf']=clf
        tup_train=[]
        tup_test=[]
        for i in cv_scores.keys():
            tup_train.append((i,cv_scores[i]))
        for i in test_scores.keys():
            tup_test.append((i,test_scores[i]))

        answer['plot_k_vs_score_train']=tup_train
        answer['plot_k_vs_score_test']=tup_test
        answer['text_rate_accuracy_change']="The model consistently shows enhanced accuracy with increasing values of k when applied to the testing data, indicating improved proficiency in predicting the top-k classes as k grows."
        answer['text_is_topk_useful_and_why']="The top_k_accuracy score is not particularly useful in this context since it doesn't offer significant insight. This metric is most effective when dealing with datasets containing two or more classes and when the dataset is balanced, which is not the case in our problem."
        
        for i in cv_scores.keys():
            if i not in answer:
                answer[i]={"score_train":cv_scores[i],"score_test":test_scores[i]}

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)


        indices_nines = np.where(y == 9)[0]
        indices_nines_test = np.where(ytest == 9)[0]
        
     
        num_to_remove = int(0.9 * len(indices_nines))
        indices_to_remove = np.random.choice(indices_nines, size=num_to_remove, replace=False)
        
  
        X = np.delete(X, indices_to_remove, axis=0)
        y = np.delete(y, indices_to_remove)
        
        # Convert 7s to 0s and 9s to 1s
        y[y == 7] = 0
        y[y == 9] = 1
        

        indices_to_remove_test = np.random.choice(indices_nines_test, size=int(0.9 * len(indices_nines_test)), replace=False)
        Xtest = np.delete(Xtest, indices_to_remove_test, axis=0)
        ytest = np.delete(ytest, indices_to_remove_test)
        
        ytest[ytest == 7] = 0
        ytest[ytest == 9] = 1
                    
        answer["length_Xtrain"] = len(X)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = X.max()
        answer["max_Xtest"] = Xtest.max()

        # Answer is a dictionary with the same keys as part 1.B

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        # Enter your code and fill the `answer` dictionary
     
        
        answer={}
        
        clf=SVC(random_state=self.seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
 
        scoring = {'f1': make_scorer(f1_score, average='macro'),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'accuracy':'accuracy'}
    
        scores_cv = {metric: cross_val_score(clf, X, y, scoring=scoring[metric], cv=cv)
          for metric in scoring}
        
        scores_cv_stra={}
        scores_cv_stra['mean_accuracy']=np.mean(scores_cv['accuracy'])
        scores_cv_stra['mean_recall']=np.mean(scores_cv['recall'])
        scores_cv_stra['mean_precision']=np.mean(scores_cv['precision'])
        scores_cv_stra['mean_f1']=np.mean(scores_cv['f1'])
        scores_cv_stra['std_accuracy']=np.std(scores_cv['accuracy'])
        scores_cv_stra['std_recall']=np.std(scores_cv['recall'])
        scores_cv_stra['std_precision']=np.std(scores_cv['precision'])
        scores_cv_stra['std_f1']=np.std(scores_cv['f1'])
        
        answer["scores"]=scores_cv_stra
        answer['cv']=cv
        answer['clf']=clf
        
        if scores_cv['precision'].mean() > scores_cv['recall'].mean():
            answer["is_precision_higher_than_recall"]=True
        else:
            answer["is_precision_higher_than_recall"]=False
            
        answer['explain_is_precision_higher_than_recall']='Precision being higher than recall indicates that the model has a higher ability to correctly identify relevant instances among the total predicted instances'
        
        clf.fit(X,y)
        y_pred_train=clf.predict(X)
        answer['confusion_matrix_train']=confusion_matrix(y_pred_train,y)
        
        y_pred_test=clf.predict(Xtest)
        answer['confusion_matrix_test']=confusion_matrix(y_pred_test,ytest)

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        

        SV_clf_weighted = SVC(random_state=self.seed, class_weight={0: class_weights[0], 1: class_weights[1]})
        

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        

        scoring = {
            'f1': make_scorer(f1_score, average='macro'),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'accuracy': 'accuracy'
        }
        
     
        evaluation_scores_weighted = {metric: cross_val_score(SV_clf_weighted, X, y, scoring=scoring[metric], cv=cv)
                                      for metric in scoring}
        

        evaluation_scores_weighted = {}
        evaluation_scores_weighted['mean_accuracy'] = np.mean(scores_cv_weighted['accuracy'])
        evaluation_scores_weighted['mean_recall'] = np.mean(scores_cv_weighted['recall'])
        evaluation_scores_weighted['mean_precision'] = np.mean(scores_cv_weighted['precision'])
        evaluation_scores_weighted['mean_f1'] = np.mean(scores_cv_weighted['f1'])
        evaluation_scores_weighted['std_accuracy'] = np.std(scores_cv_weighted['accuracy'])
        evaluation_scores_weighted['std_recall'] = np.std(scores_cv_weighted['recall'])
        evaluation_scores_weighted['std_precision'] = np.std(scores_cv_weighted['precision'])
        evaluation_scores_weighted['std_f1'] = np.std(scores_cv_weighted['f1'])
        
    
        SV_clf_weighted.fit(X, y)
        

        y_pred_train_weighted = SV_clf_weighted.predict(X)
        y_pred_test_weighted = SV_clf_weighted.predict(Xtest)
        
   
        confusion_matrix_train_weighted = confusion_matrix(y, y_pred_train_weighted)
        confusion_matrix_test_weighted = confusion_matrix(ytest, y_pred_test_weighted)
        
        answer["evaluation_scores_weighted"] = evaluation_scores_weighted
        answer['cv'] = cv
        answer['clf'] = SV_clf_weighted
        answer['class_weights'] = class_weights
        answer['confusion_matrix_train'] = confusion_matrix_train_weighted
        answer['confusion_matrix_test'] = confusion_matrix_test_weighted
        answer['explain_purpose_of_class_weights'] = "Class weights are employed to mitigate class imbalance by assigning greater penalties to misclassifications of the minority class."
        answer['explain_performance_difference'] = "The observed performance variation with class weights signifies the model's enhanced capacity to generalize to the minority class, resulting in more evenly balanced performance metrics across all classes."
        
        
        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
