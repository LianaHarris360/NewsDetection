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
  
  
  **OUTPUT**
   
   
   ![Screen Shot 2021-08-04 at 3 22 22 PM](https://user-images.githubusercontent.com/46411498/128250167-29f43a70-091e-41d4-b56e-3903e34879d6.png)

   
   
   ![Screen Shot 2021-08-04 at 3 23 09 PM](https://user-images.githubusercontent.com/46411498/128250146-30923bfa-e7f4-494f-9d35-36b069751438.png)

   
   ![Screen Shot 2021-08-04 at 3 23 46 PM](https://user-images.githubusercontent.com/46411498/128250106-0626ae0b-b191-4ee9-8ac9-970ba6ffa5e2.png)

