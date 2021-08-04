# NewsDetection
This project utilizes a Python Script to detect if news (generated from .csv file) is Real or Fake.

With the use of sklearn, the goal is to build a TfidfVectorizer on the dataset. Then, compute a PassiveAggressive Classifier and fit the model. The accuracy score and confusion matrix will tell how well the model detects fake news.

USEFUL TERMS:

TF (Term Frequency): The number of times a word appears in a document.

IDF (Inverse Document Frequency): Words that can occur many times a document, but 
   are of little importance (such as "is", "of", or "that").

Passive Aggressive algorithms are online learning algorithms. This algorithm
    remains passive for a correct classification outcome, and turns aggressive in 
    the event of a miscalculation, updating and adjusting. Its purpose is to make 
    updates that correct the loss, causing very little change in the norm of the weight vector.
    
    
   *Data-Flair Tutorial
