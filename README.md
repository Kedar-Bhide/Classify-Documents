# Document Classifier

The classify.py script attempts to classify documents as either from 2016 or 2020. A classifier is trained to determing whcih label a query document is more likely to have. A corpus (a collection of documents) is read in with two possible labels.

Here's the twist: the corpus is created from CS 540 essays about AI100 from 2016 and 2020 on the same topic. Based on the fitted model, this is an attempt to predict whether an essay was written in 2016 or 2020.

The three main steps included are: 1) Loading the data into a convenient representation 2) Computing probabilities from the constructed representation 3) Using the probabilities computed to create a prediction model and using this model to classify test data
